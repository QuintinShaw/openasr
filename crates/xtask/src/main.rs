use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

const DEFAULT_INVENTORY_RELATIVE_PATH: &str = "tooling/model-family-inventory.v1.json";
const FAMILY_MODULE_SENTINEL_BEGIN: &str = "// xtask generated model-family modules begin";
const FAMILY_MODULE_SENTINEL_END: &str = "// xtask generated model-family modules end";

#[derive(Debug, Parser)]
#[command(name = "cargo xtask", about = "OpenASR repository maintenance tasks")]
struct Cli {
    #[command(subcommand)]
    command: CommandGroup,
}

#[derive(Debug, Subcommand)]
enum CommandGroup {
    /// Scaffold a model-family package without silently registering it.
    Family {
        #[command(subcommand)]
        command: FamilyCommand,
    },
    /// Re-run the production model-pack preflight on every pack installed in
    /// the local model store. This is the shipped-pack regression gate: a
    /// contract change that would reject an already-released pack fails here
    /// before it can ship. The store itself is verified through the product
    /// read paths (refs parsed as their authoritative record type with
    /// model id / quant / sha256 / size validated against the
    /// content-addressed object, then every object re-hashed against its
    /// digest), and the preflight runs as a subprocess of the workspace
    /// `openasr` CLI -- the exact production path, sandbox included. Skips
    /// with an explicit warning when no packs are installed, unless
    /// `--require-store` turns that skip into a failure.
    VerifyInstalledPacks {
        /// Override the OpenASR home directory (defaults to `$OPENASR_HOME`,
        /// then `~/.openasr`).
        #[arg(long)]
        home: Option<PathBuf>,
        /// Fail when no installed model store exists (release gates); the
        /// default prints an explicit warning and skips.
        #[arg(long)]
        require_store: bool,
    },
}

#[derive(Debug, Subcommand)]
enum FamilyCommand {
    /// Create an intentionally incomplete model-family onboarding skeleton.
    New {
        /// Rust module directory under `crates/openasr-core/src/models/`.
        module_slug: String,
        /// Canonical lower-kebab conformance profile id. Defaults to the
        /// one-time scaffold spelling conversion from `module_slug`.
        #[arg(long)]
        profile_id: Option<String>,
    },
    /// Run weight-free structural model-family conformance gates.
    Conformance {
        /// Canonical lower-kebab profile to validate before global gates run.
        #[arg(long)]
        profile_id: Option<String>,
    },
    /// Export the Rust descriptor inventory to a deterministic JSON file.
    ExportInventory {
        /// Verify that the generated file is already up to date.
        #[arg(long)]
        check: bool,
        /// Override the generated inventory path.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let workspace_root = workspace_root();
    match cli.command {
        CommandGroup::Family { command } => match command {
            FamilyCommand::New {
                module_slug,
                profile_id,
            } => {
                let profile_id = match profile_id {
                    Some(profile_id) => {
                        validate_profile_id(&profile_id)?;
                        profile_id
                    }
                    None => profile_id_from_module_slug(&module_slug),
                };
                scaffold_family(&workspace_root, &module_slug, &profile_id)
            }
            FamilyCommand::Conformance { profile_id } => {
                run_conformance(&workspace_root, profile_id.as_deref())
            }
            FamilyCommand::ExportInventory { check, output } => {
                export_inventory(&workspace_root, output.as_deref(), check)
            }
        },
        CommandGroup::VerifyInstalledPacks {
            home,
            require_store,
        } => verify_installed_packs(&workspace_root, home.as_deref(), require_store),
    }
}

/// Resolve the OpenASR home directory through the product's own rule
/// (`$OPENASR_HOME`, then `~/.openasr`), unless the gate got an explicit
/// `--home` override.
fn resolve_openasr_home_dir(home_override: Option<&Path>) -> Result<PathBuf> {
    match home_override {
        Some(home) => Ok(home.to_path_buf()),
        None => openasr_core::openasr_home().map_err(|error| anyhow::anyhow!("{error}")),
    }
}

/// Verify every pack installed in the local model store, fail-closed, then
/// run the production CLI preflight on each. The models root is resolved
/// exactly like the product resolves it (`OPENASR_MODELS_DIR` env, then
/// `config.models_dir`, then `<home>/models`) because the store reader this
/// gate delegates to applies that rule itself.
fn verify_installed_packs(
    workspace_root: &Path,
    home_override: Option<&Path>,
    require_store: bool,
) -> Result<()> {
    let home = resolve_openasr_home_dir(home_override)?;
    let packs = check_installed_store(&home, require_store)?;
    if packs.is_empty() {
        return Ok(());
    }
    verify_store_object_bytes(&home)?;
    let cli = resolve_openasr_cli(workspace_root)?;
    let mut failures = 0usize;
    for pack in &packs {
        let label = format!("{}:{}", pack.model_id, pack.quant);
        match run_cli_preflight(&cli, &pack.path) {
            Ok(()) => println!("PASS {label} {}", pack.path.display()),
            Err(error) => {
                println!("FAIL {label} {}: {error:#}", pack.path.display());
                failures += 1;
            }
        }
    }
    println!(
        "{} installed pack(s) checked against the production CLI preflight, {failures} failed",
        packs.len()
    );
    if failures > 0 {
        bail!("{failures} installed pack(s) failed the production preflight");
    }
    Ok(())
}

/// Validate the installed store through the product's own store reader:
/// `openasr_core::InstalledModelStore` parses every ref as the authoritative
/// installed-pack record and validates model id, quant, sha256, size, and
/// the digest-derived object path (the ref's recorded path is never an
/// authority, symlinked objects are rejected). Any diagnostic fails the gate
/// -- a corrupt ref is not a SKIP. A missing or empty store skips with an
/// explicit uppercase warning, or fails outright under `--require-store`.
fn check_installed_store(
    home: &Path,
    require_store: bool,
) -> Result<Vec<openasr_core::InstalledPack>> {
    let config = openasr_core::load_config(home).unwrap_or_default();
    let models_dir = openasr_core::models_dir(home, &config);
    let refs_dir = models_dir.join("refs");
    let store = openasr_core::InstalledModelStore::read(home)
        .map_err(|error| anyhow::anyhow!("reading the installed model store failed: {error}"))?;
    if !store.diagnostics().is_empty() {
        for diagnostic in store.diagnostics() {
            println!("FAIL {}: {}", diagnostic.path.display(), diagnostic.reason);
        }
        bail!(
            "{} installed pack ref(s) failed store verification",
            store.diagnostics().len()
        );
    }
    let packs = store.into_packs();
    if packs.is_empty() {
        let models_dir_display = models_dir.display();
        let warning = if refs_dir.is_dir() {
            format!(
                "WARNING: NO INSTALLED MODEL PACKS UNDER {models_dir_display} -- \
                 SHIPPED-PACK REGRESSION GATE SKIPPED"
            )
        } else {
            format!(
                "WARNING: NO INSTALLED MODEL STORE AT {models_dir_display} -- \
                 SHIPPED-PACK REGRESSION GATE SKIPPED"
            )
        };
        if require_store {
            bail!("{warning} (--require-store was set)");
        }
        println!("{warning}");
    }
    Ok(packs)
}

/// Re-hash every object a ref names through the product's store verifier
/// (`openasr model-pack verify`'s engine), so the gate fails closed on a
/// byte-level sha/size mismatch, not just on ref structure.
fn verify_store_object_bytes(home: &Path) -> Result<()> {
    let verification = openasr_core::verify_model_store(home)
        .map_err(|error| anyhow::anyhow!("verifying installed store objects failed: {error}"))?;
    let mut failures = 0usize;
    for check in &verification.checked {
        if let Some(reason) = &check.failure {
            println!("FAIL {} ({}): {reason}", check.pull, check.digest);
            failures += 1;
        }
    }
    if failures > 0 {
        bail!("{failures} installed object(s) failed sha256/size verification");
    }
    Ok(())
}

/// Locate the workspace `openasr` CLI, building it when missing. The gate
/// must run the production binary (the sandboxed GGUF parser re-execs the
/// CLI itself); if no binary can be produced the gate errors instead of
/// degrading to any in-process path.
fn resolve_openasr_cli(workspace_root: &Path) -> Result<PathBuf> {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"));
    if let Some(cli) = find_openasr_cli_binary(&target_dir) {
        return Ok(cli);
    }
    println!(
        "openasr CLI not found under {}; building it (cargo build -p openasr-cli)",
        target_dir.display()
    );
    let status = Command::new("cargo")
        .args(["build", "-p", "openasr-cli"])
        .current_dir(workspace_root)
        .stdin(Stdio::null())
        .status()
        .context("start cargo build -p openasr-cli")?;
    if !status.success() {
        bail!("cargo build -p openasr-cli failed ({status})");
    }
    find_openasr_cli_binary(&target_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "openasr CLI binary not found under {} after building; refusing to \
             degrade the shipped-pack gate to a non-production preflight",
            target_dir.display()
        )
    })
}

/// Prefer an existing release build, then a debug build.
fn find_openasr_cli_binary(target_dir: &Path) -> Option<PathBuf> {
    let binary = format!("openasr{}", std::env::consts::EXE_SUFFIX);
    ["release", "debug"]
        .iter()
        .map(|profile| target_dir.join(profile).join(&binary))
        .find(|candidate| candidate.is_file())
}

/// Run the exact production preflight (`openasr model-pack preflight`) as a
/// subprocess of the workspace CLI. Non-zero exit fails the gate with the
/// CLI's own diagnostic output.
fn run_cli_preflight(cli: &Path, pack: &Path) -> Result<()> {
    let output = Command::new(cli)
        .args(["model-pack", "preflight"])
        .arg(pack)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("start production preflight {}", cli.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!(
        "preflight exited with {}{}{}",
        output.status,
        if stdout.trim().is_empty() {
            String::new()
        } else {
            format!("\n{stdout}", stdout = stdout.trim())
        },
        if stderr.trim().is_empty() {
            String::new()
        } else {
            format!("\n{stderr}", stderr = stderr.trim())
        },
    );
}

fn workspace_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is crates/xtask.  Keeping path discovery anchored to
    // the manifest avoids depending on the caller's current directory.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask must live two levels below the workspace root")
        .to_path_buf()
}

fn validate_family_slug(slug: &str) -> Result<()> {
    if slug.is_empty() {
        bail!("family slug must not be empty");
    }
    if slug.starts_with('_') || slug.ends_with('_') || slug.contains("__") {
        bail!(
            "family slug must be snake_case without leading, trailing, or repeated underscores: {slug}"
        );
    }
    let mut chars = slug.chars();
    let Some(first) = chars.next() else {
        bail!("family slug must not be empty");
    };
    if !first.is_ascii_lowercase() {
        bail!("family slug must start with a lowercase ASCII letter: {slug}");
    }
    if !chars.all(|character| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
    }) {
        bail!(
            "family slug must contain only lowercase ASCII letters, digits, and underscores: {slug}"
        );
    }
    if slug.split('_').any(|segment| {
        segment.is_empty() || !segment.starts_with(|character: char| character.is_ascii_lowercase())
    }) {
        bail!("each family slug segment must start with a lowercase ASCII letter: {slug}");
    }
    Ok(())
}

fn validate_profile_id(profile_id: &str) -> Result<()> {
    if profile_id.is_empty() {
        bail!("profile id must not be empty");
    }
    if profile_id.starts_with('-') || profile_id.ends_with('-') || profile_id.contains("--") {
        bail!(
            "profile id must be lower-kebab without leading, trailing, or repeated hyphens: {profile_id}"
        );
    }
    let mut chars = profile_id.chars();
    let Some(first) = chars.next() else {
        bail!("profile id must not be empty");
    };
    if !first.is_ascii_lowercase() {
        bail!("profile id must start with a lowercase ASCII letter: {profile_id}");
    }
    if !chars.all(|character| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
    }) {
        bail!(
            "profile id must contain only lowercase ASCII letters, digits, and hyphens: {profile_id}"
        );
    }
    if profile_id.split('-').any(|segment| {
        segment.is_empty() || !segment.starts_with(|character: char| character.is_ascii_lowercase())
    }) {
        bail!("each profile id segment must start with a lowercase ASCII letter: {profile_id}");
    }
    Ok(())
}

fn profile_id_from_module_slug(module_slug: &str) -> String {
    // This conversion is intentionally confined to the scaffold command. The
    // generated runtime descriptor receives the literal profile id and never
    // derives one from a module path.
    module_slug.replace('_', "-")
}

fn scaffold_family(workspace_root: &Path, module_slug: &str, profile_id: &str) -> Result<()> {
    validate_family_slug(module_slug)?;
    validate_profile_id(profile_id)?;
    let family_dir = workspace_root
        .join("crates/openasr-core/src/models")
        .join(module_slug);
    let models_path = workspace_root.join("crates/openasr-core/src/models.rs");
    if family_dir.exists() {
        bail!(
            "refusing to overwrite existing family path: {}",
            family_dir.display()
        );
    }
    let original_models = fs::read(&models_path)
        .with_context(|| format!("read authoritative module file {}", models_path.display()))?;
    let updated_models = render_models_with_family(&original_models, module_slug)?;

    fs::create_dir(&family_dir)
        .with_context(|| format!("create family directory {}", family_dir.display()))?;
    let files = [
        ("mod.rs", render_family_module(module_slug, profile_id)),
        ("README.md", render_family_readme(module_slug, profile_id)),
        (
            "architecture.rs",
            render_family_architecture(module_slug, profile_id),
        ),
        (
            "package_import.rs",
            render_family_package_import(module_slug),
        ),
        (
            "runtime_contract.rs",
            render_family_runtime_contract(module_slug),
        ),
    ];
    let mut created = Vec::new();
    for (name, contents) in files {
        let path = family_dir.join(name);
        if let Err(error) = create_new_text_file(&path, contents.as_bytes()) {
            // Only remove files this invocation created.  The directory itself
            // is removed only if it is still empty, so a concurrent/user file
            // is never touched.
            for created_path in created {
                let _ = fs::remove_file(created_path);
            }
            let _ = fs::remove_dir(&family_dir);
            return Err(error).with_context(|| format!("write family scaffold {}", path.display()));
        }
        created.push(path);
    }

    // Edit the authoritative module list only after every new file exists, and
    // replace it atomically. The re-read catches ordinary concurrent edits;
    // failures remove only files created by this invocation.
    let current_models = match fs::read(&models_path) {
        Ok(contents) => contents,
        Err(error) => {
            cleanup_created_scaffold(&family_dir, &created);
            return Err(error).with_context(|| format!("re-read {}", models_path.display()));
        }
    };
    if current_models != original_models {
        cleanup_created_scaffold(&family_dir, &created);
        bail!("authoritative module file changed while scaffolding; refusing to overwrite it");
    }
    if let Err(error) = atomic_replace(&models_path, &updated_models) {
        cleanup_created_scaffold(&family_dir, &created);
        return Err(error)
            .with_context(|| format!("wire family module in {}", models_path.display()));
    }

    println!(
        "created incomplete family scaffold at {}\n\
         module_slug={module_slug}, profile_id={profile_id}\n\
         It is wired into a compile-red sentinel; complete every facet and\n\
         register it in the authoritative descriptor before removing compile_error!.",
        family_dir.display()
    );
    Ok(())
}

fn render_models_with_family(original: &[u8], slug: &str) -> Result<Vec<u8>> {
    let source = std::str::from_utf8(original).context("models.rs is not valid UTF-8")?;
    let module_decl = format!("pub(crate) mod {slug};");
    if source.lines().any(|line| line.trim() == module_decl) {
        bail!("family module declaration already exists in models.rs: {module_decl}");
    }
    let begin = source
        .find(FAMILY_MODULE_SENTINEL_BEGIN)
        .context("models.rs is missing the family-module begin sentinel")?;
    let end = source
        .find(FAMILY_MODULE_SENTINEL_END)
        .context("models.rs is missing the family-module end sentinel")?;
    if begin >= end {
        bail!("models.rs family-module sentinels are out of order");
    }
    let end_line_start = source[..end].rfind('\n').map_or(0, |position| position + 1);
    let mut updated = Vec::with_capacity(original.len() + module_decl.len() + 1);
    updated.extend_from_slice(&original[..end_line_start]);
    if !updated.ends_with(b"\n") {
        updated.push(b'\n');
    }
    updated.extend_from_slice(module_decl.as_bytes());
    updated.push(b'\n');
    updated.extend_from_slice(&original[end_line_start..]);
    Ok(updated)
}

fn atomic_replace(path: &Path, contents: &[u8]) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("models.rs");
    let temporary_path =
        path.with_file_name(format!(".{file_name}.xtask-{}.tmp", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        file.write_all(contents)?;
        file.sync_all()?;
        replace_existing_file(&temporary_path, path)?;
        sync_parent_directory(path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

#[cfg(not(windows))]
fn replace_existing_file(replacement: &Path, replaced: &Path) -> io::Result<()> {
    fs::rename(replacement, replaced)
}

#[cfg(windows)]
fn replace_existing_file(replacement: &Path, replaced: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    use windows_sys::Win32::Storage::FileSystem::{REPLACEFILE_IGNORE_MERGE_ERRORS, ReplaceFileW};

    let replaced = replaced
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let replacement = replacement
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // SAFETY: both UTF-16 buffers are NUL-terminated and live for the call;
    // backup, exclude, and reserved pointers are explicitly unused.
    let succeeded = unsafe {
        ReplaceFileW(
            replaced.as_ptr(),
            replacement.as_ptr(),
            ptr::null(),
            REPLACEFILE_IGNORE_MERGE_ERRORS,
            ptr::null(),
            ptr::null(),
        )
    };
    if succeeded == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(windows)]
fn sync_parent_directory(_path: &Path) -> io::Result<()> {
    // ReplaceFileW atomically swaps the directory entry. Opening directories
    // for an additional durability flush needs platform-specific privileges
    // and is not required for the scaffold's fail-closed registration guard.
    Ok(())
}

fn cleanup_created_scaffold(family_dir: &Path, created: &[PathBuf]) {
    for path in created {
        let _ = fs::remove_file(path);
    }
    let _ = fs::remove_dir(family_dir);
}

fn create_new_text_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(contents)?;
    file.sync_all()
}

fn render_family_module(module_slug: &str, profile_id: &str) -> String {
    format!(
        "//! Incomplete model-family onboarding scaffold for `{module_slug}`.\n//!\n//! `cargo xtask family new` wires this module into the stable compile-red\n//! sentinel in `models.rs`; it must never silently reach runtime.\n\n#[allow(unused)]\npub(crate) mod architecture;\n#[allow(unused)]\npub(crate) mod package_import;\n#[allow(unused)]\npub(crate) mod runtime_contract;\n\ncompile_error!(\"family `{module_slug}` (profile `{profile_id}`) is incomplete: fill every descriptor facet, pack/runtime/conformance test, and shared execution seam before registration\");\n"
    )
}

fn render_family_readme(module_slug: &str, profile_id: &str) -> String {
    let mut readme = format!(
        "# {module_slug} model family scaffold\n\nGenerated by `cargo xtask family new {module_slug} --profile-id {profile_id}`.\nThe Rust module path (`{module_slug}`) and canonical conformance profile\n(`{profile_id}`) are separate identifiers. The generated profile is a literal\nin the eventual descriptor; runtime code must not derive it from a path.\n\nThis is an intentionally incomplete skeleton. `mod.rs` contains a\nfail-closed `compile_error!` until the family is genuinely wired. The scaffold\nprovides source files and an implementation checklist, not a runnable model or\na fake descriptor/quantization contract.\n\n## Required before registration\n\n- Fill all seven facets in `architecture.rs` and register the complete row in\n  the authoritative Rust inventory.\n- Implement the real `PackEnvelope`/`OasrPackWriter` importer and a concrete\n  `TENSOR_QUANTIZATION_CONTRACT`; raw GGUF output is not a production importer.\n- Implement the runtime validator in `runtime_contract.rs`, then add positive\n  and negative fixtures for `PackVerifier`.\n- Reuse shared decode drivers, graph blocks, cancellation fences, and runtime\n  admission. Add a narrow adapter only for mathematical differences.\n- Run the weight-free structural command:\n  `cargo xtask family conformance --profile-id {profile_id}`. This checks source,\n  registry, pack, quantization, inventory, Python projection, regeneration,\n  and the static GPU-placement gate; real backend smoke and benchmark receipts\n  remain release/manual obligations.\n\nReference migration order: FunASR-Nano -> Parakeet-CTC -> Qwen3-ASR ->\nParakeet-TDT. Remove `compile_error!` only in the same change that wires the\ncomplete family.\n"
    );
    readme.push_str(
        "\n## Integration scope\n\nChoose and record exactly one scope: `core-only`, `staged release candidate`,\n\
         or `public-ready`. Staged/public-ready work must follow\n\
         `docs/MODEL_ONBOARDING.md`, Step 5, for catalog inputs, real-weight\n\
         receipts, family audit, and regression coverage. Never fabricate\n\
         publication metadata or hand-edit generated registry/catalog files.\n",
    );
    readme
}

fn render_family_architecture(module_slug: &str, profile_id: &str) -> String {
    format!(
        "//! Descriptor skeleton for `{module_slug}` (profile `{profile_id}`).\n//!\n//! These `todo!` expressions are deliberate: they make every required facet\n//! visible without inventing a self-asserted contract. Replace each one with\n//! the reviewed, concrete value before removing the fail-closed sentinel.\n\nuse crate::arch::OpenAsrArchitectureDescriptor;\n\n#[allow(dead_code)]\npub(crate) fn architecture_descriptor() -> OpenAsrArchitectureDescriptor {{\n    OpenAsrArchitectureDescriptor {{\n        identity: todo!(\"fill identity facet for module `{module_slug}`\"),\n        pack_contract: todo!(\"fill pack_contract facet for module `{module_slug}`\"),\n        execution_contract: todo!(\"fill execution_contract facet for module `{module_slug}`\"),\n        topology_contract: todo!(\"fill topology_contract facet for module `{module_slug}`\"),\n        optimization_contract: todo!(\"fill optimization_contract facet for module `{module_slug}`\"),\n        quantization_contract: todo!(\"fill quantization_contract facet for module `{module_slug}`\"),\n        conformance_contract: todo!(\"fill conformance_contract facet with literal profile_id `{profile_id}`\"),\n    }}\n}}\n"
    )
}

fn render_family_package_import(module_slug: &str) -> String {
    format!(
        "//! Pack-import implementation checklist for `{module_slug}`.\n//!\n//! This file intentionally contains no placeholder contract. Keep the\n//! imports below only while implementing the real family importer; replace\n//! this checklist with concrete code before removing `compile_error!` in\n//! `mod.rs`.\n\n#[allow(unused_imports)]\nuse crate::models::oasr_metadata::{{OasrPackWriter, PackEnvelope}};\n#[allow(unused_imports)]\nuse crate::models::pack_quant::TensorQuantizationContract;\n\n// Required concrete seams:\n// 1. PackEnvelope::asr(...) or PackEnvelope::aux(...) selects the reviewed route.\n// 2. OasrPackWriter::write(...) / begin_repack(...) emits transactionally and\n//    returns the exact PackVerifier-produced VerifiedPack.\n// 3. Declare `pub(crate) const TENSOR_QUANTIZATION_CONTRACT` with a real\n//    family-owned mapping; do not add a default, wildcard, or fake TODO value.\n// 4. Add source-shape/tensor-role tests and both valid and invalid pack fixtures.\n\n#[allow(dead_code)]\nfn shared_pack_seams_are_in_scope(\n    envelope: PackEnvelope,\n    _writer: OasrPackWriter,\n    _quantization: TensorQuantizationContract,\n) -> PackEnvelope {{\n    envelope\n}}\n"
    )
}

fn render_family_runtime_contract(module_slug: &str) -> String {
    format!(
        "//! Runtime-pack validator skeleton for `{module_slug}`.\n//!\n//! The fixed error is intentional: a family cannot become admissible while\n//! its real metadata/tensor validator is absent.\n\n#[allow(dead_code)]\npub(crate) fn validate_runtime_pack_contract(\n    _preflight: &crate::GgufRuntimeSourcePreflight,\n) -> Result<(), String> {{\n    Err(\"family `{module_slug}` runtime contract is incomplete\".to_string())\n}}\n"
    )
}

fn inventory_contains_profile(profile_id: &str) -> bool {
    openasr_core::builtin_model_family_inventory()
        .families
        .iter()
        .any(|family| family.conformance.profile_id == profile_id)
}

fn run_checked_command(
    workspace_root: &Path,
    program: &Path,
    args: &[&str],
    label: &str,
) -> Result<()> {
    let status = Command::new(program)
        .current_dir(workspace_root)
        .args(args)
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("start structural conformance command: {label}"))?;
    if !status.success() {
        bail!("structural conformance command failed: {label} ({status})");
    }
    Ok(())
}

fn run_conformance(workspace_root: &Path, profile_id: Option<&str>) -> Result<()> {
    if let Some(profile_id) = profile_id {
        validate_profile_id(profile_id)?;
        if !inventory_contains_profile(profile_id) {
            bail!(
                "unknown conformance profile id `{profile_id}`; it must already exist in the Rust builtin inventory"
            );
        }
        println!("running weight-free structural conformance gates (profile: {profile_id})");
    } else {
        println!("running weight-free structural conformance gates (all builtin profiles)");
    }

    // The profile is a diagnostic selector only. Structural gates remain
    // global because generated projections and shared admission are global
    // invariants. This command deliberately does not run real weights,
    // backend inference, or benchmarks; those are release/manual C-class
    // obligations with benchmark receipts.
    export_inventory(workspace_root, None, true)?;

    run_checked_command(
        workspace_root,
        Path::new("cargo"),
        &["nextest", "run", "-p", "openasr-core", "--lib"],
        "cargo nextest run -p openasr-core --lib",
    )?;
    run_checked_command(
        workspace_root,
        Path::new("python3"),
        &[
            "-m",
            "unittest",
            "discover",
            "-s",
            "tooling/publish-model/scripts",
            "-p",
            "*_test.py",
        ],
        "python3 -m unittest discover -s tooling/publish-model/scripts -p '*_test.py'",
    )?;
    run_checked_command(
        workspace_root,
        Path::new("tooling/publish-model/scripts/regenerate_all.sh"),
        &["--check"],
        "tooling/publish-model/scripts/regenerate_all.sh --check",
    )?;
    run_checked_command(
        workspace_root,
        Path::new("scripts/gpu-weight-placement-gate.sh"),
        &[],
        "scripts/gpu-weight-placement-gate.sh",
    )?;
    Ok(())
}

fn export_inventory(workspace_root: &Path, output: Option<&Path>, check: bool) -> Result<()> {
    let output_path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| workspace_root.join(DEFAULT_INVENTORY_RELATIVE_PATH));
    let rendered = render_inventory()?;
    if check {
        let existing = fs::read(&output_path).with_context(|| {
            format!(
                "read generated inventory for --check: {}",
                output_path.display()
            )
        })?;
        if existing != rendered.as_bytes() {
            bail!(
                "generated model-family inventory is out of date: {}; run `cargo xtask family export-inventory`",
                output_path.display()
            );
        }
        println!(
            "model-family inventory is up to date: {}",
            output_path.display()
        );
        return Ok(());
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create inventory parent {}", parent.display()))?;
    }
    let mut file = File::create(&output_path)
        .with_context(|| format!("create inventory {}", output_path.display()))?;
    file.write_all(rendered.as_bytes())?;
    file.sync_all()?;
    println!("wrote model-family inventory: {}", output_path.display());
    Ok(())
}

fn render_inventory() -> Result<String> {
    let inventory = openasr_core::builtin_model_family_inventory();
    let mut rendered = serde_json::to_string_pretty(&inventory).context("serialize inventory")?;
    rendered.push('\n');
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn accepts_normal_snake_case_slugs() {
        for slug in [
            "qwen3_asr",
            "parakeet_ctc",
            "moss_transcribe_diarize",
            "family2",
        ] {
            validate_family_slug(slug).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_snake_case_slugs() {
        for slug in [
            "",
            "Qwen",
            "qwen-asr",
            "qwen__asr",
            "_qwen",
            "qwen_",
            "qwen_2x",
        ] {
            assert!(
                validate_family_slug(slug).is_err(),
                "accepted invalid slug {slug}"
            );
        }
    }

    #[test]
    fn accepts_normal_lower_kebab_profile_ids() {
        for profile_id in ["qwen", "parakeet-tdt", "moss-transcribe-diarize", "family2"] {
            validate_profile_id(profile_id).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_lower_kebab_profile_ids() {
        for profile_id in [
            "",
            "Qwen",
            "qwen_asr",
            "qwen--asr",
            "-qwen",
            "qwen-",
            "qwen-2x",
        ] {
            assert!(
                validate_profile_id(profile_id).is_err(),
                "accepted invalid profile id {profile_id}"
            );
        }
    }

    #[test]
    fn module_slug_profile_conversion_is_scaffold_only() {
        assert_eq!(
            profile_id_from_module_slug("moss_transcribe_diarize"),
            "moss-transcribe-diarize"
        );
    }

    #[test]
    fn scaffold_refuses_overwrite() {
        let root = tempdir().unwrap();
        let model_root = root.path().join("crates/openasr-core/src/models");
        fs::create_dir_all(model_root.join("existing")).unwrap();
        assert!(scaffold_family(root.path(), "existing", "existing").is_err());
        assert!(model_root.join("existing").is_dir());
    }

    #[test]
    fn scaffold_wires_compile_error_module_inside_sentinel() {
        let root = tempdir().unwrap();
        let core_root = root.path().join("crates/openasr-core");
        fs::create_dir_all(core_root.join("src/models")).unwrap();
        let models_path = core_root.join("src/models.rs");
        fs::write(
            &models_path,
            format!(
                "pub(crate) mod existing;\n\n{FAMILY_MODULE_SENTINEL_BEGIN}\n{FAMILY_MODULE_SENTINEL_END}\n"
            ),
        )
        .unwrap();

        scaffold_family(root.path(), "sample_family", "sample-family").unwrap();

        let models = fs::read_to_string(models_path).unwrap();
        let declaration = "pub(crate) mod sample_family;";
        assert!(models.contains(declaration));
        let begin = models.find(FAMILY_MODULE_SENTINEL_BEGIN).unwrap();
        let end = models.find(FAMILY_MODULE_SENTINEL_END).unwrap();
        let declaration_offset = models.find(declaration).unwrap();
        assert!(begin < declaration_offset && declaration_offset < end);
        let family_dir = core_root.join("src/models/sample_family");
        assert!(
            fs::read_to_string(family_dir.join("mod.rs"))
                .unwrap()
                .contains("pub(crate) mod architecture;")
        );
        assert!(
            fs::read_to_string(family_dir.join("mod.rs"))
                .unwrap()
                .contains("compile_error!")
        );
        assert!(
            fs::read_to_string(family_dir.join("architecture.rs"))
                .unwrap()
                .contains("literal profile_id `sample-family`")
        );
        assert!(
            fs::read_to_string(family_dir.join("runtime_contract.rs"))
                .unwrap()
                .contains("runtime contract is incomplete")
        );
        assert!(
            fs::read_to_string(family_dir.join("package_import.rs"))
                .unwrap()
                .contains("TENSOR_QUANTIZATION_CONTRACT")
        );
        let readme = fs::read_to_string(family_dir.join("README.md")).unwrap();
        assert!(readme.contains("Choose and record exactly one scope"));
        assert!(readme.contains("docs/MODEL_ONBOARDING.md"));
        assert!(!family_dir.join("contract.toml").exists());
    }

    #[test]
    fn scaffold_without_sentinel_preserves_models_file_and_creates_nothing() {
        let root = tempdir().unwrap();
        let core_root = root.path().join("crates/openasr-core");
        fs::create_dir_all(core_root.join("src/models")).unwrap();
        let models_path = core_root.join("src/models.rs");
        let original = b"pub(crate) mod existing;\n";
        fs::write(&models_path, original).unwrap();

        assert!(scaffold_family(root.path(), "sample_family", "sample-family").is_err());
        assert_eq!(fs::read(&models_path).unwrap(), original);
        assert!(!core_root.join("src/models/sample_family").exists());
    }

    #[test]
    fn inventory_render_is_deterministic_and_has_trailing_newline() {
        let first = render_inventory().unwrap();
        let second = render_inventory().unwrap();
        assert_eq!(first, second);
        assert!(first.ends_with("}\n"));
    }

    #[test]
    fn export_check_detects_drift() {
        let root = tempdir().unwrap();
        let output = root.path().join("inventory.json");
        export_inventory(root.path(), Some(&output), false).unwrap();
        export_inventory(root.path(), Some(&output), true).unwrap();
        fs::write(&output, b"{}\n").unwrap();
        assert!(export_inventory(root.path(), Some(&output), true).is_err());
    }

    // --- verify-installed-packs behavior tests ------------------------------
    //
    // Each test builds a temporary store fixture. nextest runs every test in
    // its own process, so removing the models-dir env override inside a test
    // is sound and keeps the product resolution (env > config > default)
    // pointed at the fixture.

    fn clear_models_dir_env() {
        // SAFETY: nextest process isolation; single-threaded test setup.
        unsafe { std::env::remove_var(openasr_core::OPENASR_MODELS_DIR_ENV) };
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Write a content-addressed object plus its InstalledPack ref, the way
    /// the product store lays them out. Returns the ref path.
    fn write_installed_pack(
        home: &Path,
        model_id: &str,
        quant: &str,
        object_bytes: &[u8],
    ) -> PathBuf {
        let digest = sha256_hex(object_bytes);
        write_installed_pack_with_digest(home, model_id, quant, object_bytes, &digest)
    }

    fn write_installed_pack_with_digest(
        home: &Path,
        model_id: &str,
        quant: &str,
        object_bytes: &[u8],
        digest: &str,
    ) -> PathBuf {
        let object = home
            .join("models/objects/sha256")
            .join(digest)
            .join("content");
        fs::create_dir_all(object.parent().unwrap()).unwrap();
        fs::write(&object, object_bytes).unwrap();
        let record = serde_json::json!({
            "model_id": model_id,
            "display_name": model_id,
            "quant": quant,
            "suffix": "q8",
            "pull": format!("{model_id}:q8"),
            "filename": format!("{model_id}-{quant}.oasr"),
            "path": object.display().to_string(),
            "url": "https://example.invalid/model.oasr",
            "hf_revision": "test",
            "sha256": digest,
            "size_bytes": object_bytes.len(),
            "installed_at_unix_seconds": 1_u64,
        });
        let ref_path = home
            .join("models/refs")
            .join(model_id)
            .join(format!("{quant}.json"));
        fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
        fs::write(&ref_path, record.to_string()).unwrap();
        ref_path
    }

    #[test]
    fn check_installed_store_accepts_a_valid_ref_and_resolves_the_object_path() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");

        let packs = check_installed_store(home.path(), true).expect("valid store must pass");
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].model_id, "example-model");
        assert_eq!(packs[0].quant, "q8_0");
        assert_eq!(
            packs[0].path,
            home.path()
                .join("models/objects/sha256")
                .join(sha256_hex(b"pack-bytes"))
                .join("content"),
            "the gate must resolve the object from the digest, never the recorded path"
        );
    }

    #[test]
    fn check_installed_store_fails_closed_on_a_corrupt_ref_json() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        let ref_path = write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");
        fs::write(&ref_path, "{ not json").unwrap();

        let error = check_installed_store(home.path(), false)
            .expect_err("a corrupt ref must fail the gate, not skip");
        assert!(
            error.to_string().contains("failed store verification"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn check_installed_store_fails_closed_when_the_ref_has_no_path() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        let ref_path = write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");
        let mut record: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ref_path).unwrap()).unwrap();
        record
            .as_object_mut()
            .unwrap()
            .remove("path")
            .expect("ref carries a path");
        fs::write(&ref_path, record.to_string()).unwrap();

        assert!(
            check_installed_store(home.path(), false).is_err(),
            "a ref without a pack path must fail the gate, not skip"
        );
    }

    #[test]
    fn check_installed_store_fails_closed_when_the_object_is_missing() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");
        let object = home
            .path()
            .join("models/objects/sha256")
            .join(sha256_hex(b"pack-bytes"))
            .join("content");
        fs::remove_file(object).unwrap();

        assert!(
            check_installed_store(home.path(), false).is_err(),
            "a ref whose object vanished must fail the gate"
        );
    }

    #[test]
    fn check_installed_store_fails_closed_on_a_size_mismatch() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        let ref_path = write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");
        let mut record: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ref_path).unwrap()).unwrap();
        record["size_bytes"] = serde_json::json!(999_u64);
        fs::write(&ref_path, record.to_string()).unwrap();

        assert!(
            check_installed_store(home.path(), false).is_err(),
            "a ref whose size disagrees with the object must fail the gate"
        );
    }

    #[test]
    fn verify_store_object_bytes_fails_closed_on_a_sha_mismatch() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        // A valid-format digest naming bytes that do not hash to it.
        write_installed_pack_with_digest(
            home.path(),
            "example-model",
            "q8_0",
            b"tampered-bytes",
            &sha256_hex(b"original-bytes"),
        );

        let error = verify_store_object_bytes(home.path())
            .expect_err("a sha256 mismatch must fail the gate");
        assert!(
            error
                .to_string()
                .contains("failed sha256/size verification"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn verify_store_object_bytes_accepts_an_intact_object() {
        clear_models_dir_env();
        let home = tempdir().unwrap();
        write_installed_pack(home.path(), "example-model", "q8_0", b"pack-bytes");

        verify_store_object_bytes(home.path()).expect("intact object must verify");
    }

    #[test]
    fn missing_store_skips_with_a_warning_by_default_and_fails_with_require_store() {
        clear_models_dir_env();
        let home = tempdir().unwrap();

        let packs =
            check_installed_store(home.path(), false).expect("an absent store skips by default");
        assert!(packs.is_empty());

        let error = check_installed_store(home.path(), true)
            .expect_err("--require-store must fail on an absent store");
        assert!(
            error.to_string().contains("--require-store"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn find_openasr_cli_binary_prefers_release_then_debug() {
        let target = tempdir().unwrap();
        let binary = format!("openasr{}", std::env::consts::EXE_SUFFIX);
        assert!(find_openasr_cli_binary(target.path()).is_none());

        let debug = target.path().join("debug").join(&binary);
        fs::create_dir_all(debug.parent().unwrap()).unwrap();
        fs::write(&debug, b"").unwrap();
        assert_eq!(
            find_openasr_cli_binary(target.path()).unwrap(),
            debug,
            "a debug build is found when no release build exists"
        );

        let release = target.path().join("release").join(&binary);
        fs::create_dir_all(release.parent().unwrap()).unwrap();
        fs::write(&release, b"").unwrap();
        assert_eq!(
            find_openasr_cli_binary(target.path()).unwrap(),
            release,
            "a release build wins over a debug build"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_cli_preflight_reports_the_subprocess_outcome() {
        let dir = tempdir().unwrap();
        let ok_stub = dir.path().join("preflight-ok.sh");
        fs::write(&ok_stub, "#!/bin/sh\nexit 0\n").unwrap();
        let mut permissions = fs::metadata(&ok_stub).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o755);
        fs::set_permissions(&ok_stub, permissions).unwrap();
        run_cli_preflight(&ok_stub, Path::new("/unused/pack.oasr"))
            .expect("a zero exit reports success");

        let fail_stub = dir.path().join("preflight-fail.sh");
        fs::write(
            &fail_stub,
            "#!/bin/sh\necho 'pack failed the contract' >&2\nexit 1\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&fail_stub).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fail_stub, permissions).unwrap();
        let error = run_cli_preflight(&fail_stub, Path::new("/unused/pack.oasr"))
            .expect_err("a non-zero exit must fail the gate");
        assert!(
            error.to_string().contains("pack failed the contract"),
            "the CLI diagnostic must surface: {error:#}"
        );
    }
}
