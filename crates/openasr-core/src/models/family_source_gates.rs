//! Narrow source-shape CI gates for model-family trust boundaries.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprCall, ExprMethodCall, ExprStruct, Fields, GenericArgument, ImplItemFn,
    Item, ItemFn, ItemImpl, ItemMod, ItemUse, Member, Meta, PathArguments, ReturnType, Token, Type,
    Visibility,
};

fn models_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models")
}

fn parse_source(path: &Path) -> syn::File {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read production source {}: {error}", path.display()));
    syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("parse production source {}: {error}", path.display()))
}

fn assert_production_does_not_reference(path: &Path, symbol: &str) {
    let syntax = ProductionSyntax::collect(path);
    assert!(
        !syntax
            .identifiers
            .iter()
            .any(|identifier| identifier == symbol || identifier.ends_with(symbol)),
        "production source {} must derive family behavior from the architecture inventory, not reference {symbol}",
        path.display()
    );
}

fn assert_tuple_alias_components(path: &Path, alias: &str, expected: &[&str]) {
    let file = parse_source(path);
    let item = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Type(item) if item.ident == alias => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{} must declare type alias {alias}", path.display()));
    let Type::Tuple(tuple) = item.ty.as_ref() else {
        panic!("{}::{alias} must be a tuple alias", path.display());
    };
    let actual = tuple
        .elems
        .iter()
        .map(|component| match component {
            Type::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_else(|| "<empty-path>".to_string()),
            _ => panic!(
                "{}::{alias} contains a non-path key component",
                path.display()
            ),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        expected,
        "{}::{alias} must contain immutable resident-runtime identity only; request capacity belongs to session state",
        path.display()
    );
}

#[derive(Default)]
pub(super) struct ProductionSyntax {
    identifiers: BTreeSet<String>,
    calls: BTreeSet<String>,
    methods: BTreeSet<String>,
    unsafe_impl_traits: BTreeSet<String>,
    provider_name_parses: BTreeSet<String>,
    block_stack_none: bool,
    creates_request_output_pack: bool,
}

impl ProductionSyntax {
    pub(super) fn collect(path: &Path) -> Self {
        let file = parse_source(path);
        let mut syntax = Self::default();
        syntax.visit_file(&file);
        syntax
    }

    pub(super) fn references_identifier(&self, identifier: &str) -> bool {
        self.identifiers.contains(identifier)
    }

    pub(super) fn calls_or_invokes_method(&self, function: &str) -> bool {
        self.calls.contains(function) || self.methods.contains(function)
    }

    pub(super) fn has_unsafe_impl_for(&self, trait_name: &str) -> bool {
        self.unsafe_impl_traits.contains(trait_name)
    }

    fn parses_provider_names(&self) -> bool {
        !self.provider_name_parses.is_empty()
    }
}

impl<'ast> Visit<'ast> for ProductionSyntax {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        if !is_test_only(&node.attrs) {
            visit::visit_item_mod(self, node);
        }
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if !is_test_only(&node.attrs) {
            visit::visit_item_fn(self, node);
        }
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        if !is_test_only(&node.attrs) {
            if node.unsafety.is_some()
                && let Some((_, trait_path, _)) = &node.trait_
                && let Some(trait_name) = trait_path.segments.last()
            {
                self.unsafe_impl_traits.insert(trait_name.ident.to_string());
            }
            visit::visit_item_impl(self, node);
        }
    }

    fn visit_item_use(&mut self, _node: &'ast ItemUse) {
        // Imports alone do not prove that production code uses a contract or
        // primitive. Any real type, value, or call site is visited separately.
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if !is_test_only(&node.attrs) {
            visit::visit_impl_item_fn(self, node);
        }
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.identifiers.extend(
            path.segments
                .iter()
                .map(|segment| segment.ident.to_string()),
        );
        visit::visit_path(self, path);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(function) = node.func.as_ref()
            && let Some(last) = function.path.segments.last()
        {
            self.calls.insert(last.ident.to_string());
            if last.ident == "create"
                && function
                    .path
                    .segments
                    .iter()
                    .any(|segment| segment.ident == "File")
                && node.args.iter().any(expr_is_request_output_pack)
            {
                self.creates_request_output_pack = true;
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.methods.insert(node.method.to_string());
        if matches!(
            node.method.to_string().as_str(),
            "contains" | "starts_with" | "ends_with" | "eq_ignore_ascii_case"
        ) {
            for argument in &node.args {
                if let Expr::Lit(literal) = argument
                    && let syn::Lit::Str(value) = &literal.lit
                    && matches!(
                        value.value().to_ascii_lowercase().as_str(),
                        "cpu" | "gpu" | "metal" | "hip" | "rocm" | "cuda" | "nvidia" | "vulkan"
                    )
                {
                    self.provider_name_parses.insert(value.value());
                }
            }
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast ExprStruct) {
        self.block_stack_none |= node.fields.iter().any(|field| {
            matches!(&field.member, Member::Named(name) if name == "block_stack")
                && matches!(&field.expr, Expr::Path(path) if path.path.is_ident("None"))
        });
        visit::visit_expr_struct(self, node);
    }
}

fn expr_is_request_output_pack(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(reference) => expr_is_request_output_pack(&reference.expr),
        Expr::Field(field) => {
            matches!(&field.member, Member::Named(name) if name == "output_pack")
                && matches!(field.base.as_ref(), Expr::Path(path) if path.path.is_ident("request"))
        }
        _ => false,
    }
}

fn is_test_only(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attribute| {
        if attribute.path().is_ident("test") {
            return true;
        }
        if !attribute.path().is_ident("cfg") {
            return false;
        }
        let Meta::List(list) = &attribute.meta else {
            return false;
        };
        list.parse_args::<Meta>()
            .is_ok_and(|predicate| cfg_predicate_is_test_only(&predicate))
    })
}

fn cfg_predicate_is_test_only(predicate: &Meta) -> bool {
    match predicate {
        Meta::Path(path) => path.is_ident("test"),
        Meta::NameValue(value) => {
            value.path.is_ident("feature")
                && matches!(
                    &value.value,
                    Expr::Lit(literal)
                        if matches!(&literal.lit, syn::Lit::Str(feature) if feature.value() == "testing")
                )
        }
        Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            let Ok(children) =
                list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return false;
            };
            if list.path.is_ident("all") {
                children.iter().any(cfg_predicate_is_test_only)
            } else {
                !children.is_empty() && children.iter().all(cfg_predicate_is_test_only)
            }
        }
        Meta::List(_) => false,
    }
}

fn rust_files_below(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read model source directory") {
        let path = entry.expect("read model source entry").path();
        if path.is_dir() {
            rust_files_below(&path, output);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            output.push(path);
        }
    }
}

fn result_ok_type_name(output: &ReturnType) -> Option<String> {
    let ReturnType::Type(_, output) = output else {
        return None;
    };
    let Type::Path(result) = output.as_ref() else {
        return None;
    };
    let result = result.path.segments.last()?;
    if result.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &result.arguments else {
        return None;
    };
    let GenericArgument::Type(Type::Path(ok_type)) = arguments.args.first()? else {
        return None;
    };
    ok_type
        .path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn struct_carries_verified_pack(item: &syn::ItemStruct) -> bool {
    let Fields::Named(fields) = &item.fields else {
        return false;
    };
    fields.named.iter().any(|field| {
        field.ident.as_ref().is_some_and(|name| name == "verified_pack")
            && matches!(
                &field.ty,
                Type::Path(path)
                    if path.path.segments.last().is_some_and(|segment| segment.ident == "VerifiedPack")
            )
    })
}

#[test]
fn production_model_importers_cannot_call_the_raw_gguf_writer() {
    let root = models_root();
    let mut files = Vec::new();
    rust_files_below(&root, &mut files);
    let mut violations = Vec::new();
    for path in files {
        let relative = path.strip_prefix(&root).unwrap_or(&path);
        if matches!(
            relative.to_str(),
            Some("oasr_metadata.rs" | "pack_verifier.rs" | "family_source_gates.rs")
        ) {
            continue;
        }
        let syntax = ProductionSyntax::collect(&path);
        if syntax.calls.contains("write_gguf_file_v0") {
            violations.push(relative.display().to_string());
        }
    }
    assert!(
        violations.is_empty(),
        "production model importers must use OasrPackWriter; raw GGUF calls found in: {}",
        violations.join(", ")
    );
}

#[test]
fn public_runtime_pack_imports_carry_the_writer_proof() {
    let root = models_root();
    let mut files = Vec::new();
    rust_files_below(&root, &mut files);
    let mut checked = 0usize;
    let mut violations = Vec::new();

    for path in files {
        let file = parse_source(&path);
        let structs = file
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Struct(item) if !is_test_only(&item.attrs) => {
                    Some((item.ident.to_string(), item))
                }
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>();

        for function in file.items.iter().filter_map(|item| match item {
            Item::Fn(function) if !is_test_only(&function.attrs) => Some(function),
            _ => None,
        }) {
            let name = function.sig.ident.to_string();
            let is_public_pack_import = matches!(function.vis, Visibility::Public(_))
                && name.ends_with("_to_runtime_pack")
                && (name.starts_with("convert_local_") || name.starts_with("import_"));
            if !is_public_pack_import {
                continue;
            }
            checked += 1;
            let Some(ok_type) = result_ok_type_name(&function.sig.output) else {
                violations.push(format!(
                    "{}::{name} must return Result<VerifiedPack or a local result struct, _>",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
                continue;
            };
            if ok_type == "VerifiedPack" {
                continue;
            }
            if !structs
                .get(&ok_type)
                .is_some_and(|result| struct_carries_verified_pack(result))
            {
                violations.push(format!(
                    "{}::{name} returns {ok_type} without a named VerifiedPack field",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        checked > 0,
        "runtime-pack importer gate matched no functions"
    );
    assert!(
        violations.is_empty(),
        "public runtime-pack importers must return the writer's proof; output paths are diagnostic only:\n{}",
        violations.join("\n")
    );
}

#[test]
fn byte_preserving_hymt2_repack_stays_inside_the_oasr_transaction() {
    let path = models_root().join("hymt2/package_import.rs");
    let syntax = ProductionSyntax::collect(&path);
    for required in ["begin_repack", "staging_path", "commit"] {
        assert!(
            syntax.calls.contains(required) || syntax.methods.contains(required),
            "Hy-MT2's byte-preserving writer must retain transaction step {required}"
        );
    }
    assert!(
        !syntax.creates_request_output_pack,
        "custom repackers must never write the final output path directly"
    );
}

#[test]
fn removed_family_architecture_apis_cannot_return() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files_below(&root, &mut files);
    let forbidden = [
        "OpenAsrFamilyIntegrationDescriptor",
        "GgmlAsrRuntimeSourcePreflight",
        "FamilyDefinitionRegistry",
        "GgmlFamilyRegistry",
        "ggml_family_registry",
        "BUILTIN_COMPONENT_DESCRIPTORS",
        "_runtime_descriptor_v1",
        "materialize_builtin_executor_component",
        "shared_decode_driver",
        "OpenAsrExecutorOwnership",
        "OpenAsrPreparedRuntimeEviction",
        "OpenAsrGraphReuse",
        "AcousticEncoderPrefixesV1",
        "QuantComponent",
        "supports_lora_adapter",
        "with_whisper_non_streaming_cpu",
        "block_stack: None",
    ];
    let mut violations = Vec::new();
    for path in files {
        if path.ends_with("models/family_source_gates.rs") {
            continue;
        }
        let syntax = ProductionSyntax::collect(&path);
        for symbol in forbidden {
            let found = if symbol == "block_stack: None" {
                syntax.block_stack_none
            } else {
                syntax
                    .identifiers
                    .iter()
                    .any(|identifier| identifier == symbol || identifier.ends_with(symbol))
            };
            if found {
                violations.push(format!(
                    "{} contains {symbol}",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "obsolete model-family APIs are forbidden:\n{}",
        violations.join("\n")
    );
}

#[test]
fn retired_family_apis_cannot_return_to_agent_guidance() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("openasr-core lives under <repo>/crates");
    let guidance = [
        repo_root.join("AGENTS.md"),
        repo_root.join("docs/MODEL_ONBOARDING.md"),
        repo_root.join("docs/design/model-onboarding-contract.md"),
        repo_root.join("docs/design/model-family-lifecycle.md"),
    ];
    let forbidden = [
        "OpenAsrFamilyIntegrationDescriptor",
        "GgmlAsrRuntimeSourcePreflight",
        "FamilyDefinitionRegistry",
        "GgmlFamilyRegistry",
        "ggml_family_registry",
        "BUILTIN_COMPONENT_DESCRIPTORS",
        "_runtime_descriptor_v1",
        "materialize_builtin_executor_component",
        "shared_decode_driver",
        "block_stack: None",
    ];
    let mut violations = Vec::new();
    for path in guidance {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for symbol in forbidden {
            if source.contains(symbol) {
                violations.push(format!("{} contains {symbol}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "agent guidance must not teach retired model-family APIs:\n{}",
        violations.join("\n")
    );
}

#[test]
fn shared_runtime_registries_do_not_reintroduce_family_architecture_matches() {
    let root = models_root();
    for relative in [
        "runtime_prepared_registry.rs",
        "runtime_weight_component_registry.rs",
    ] {
        assert_production_does_not_reference(&root.join(relative), "_GGML_ARCHITECTURE_ID");
    }
    assert_production_does_not_reference(
        &root.join("runtime_weight_component_registry.rs"),
        "OpenAsrArchitectureRegistry",
    );
}

#[test]
fn production_family_policy_does_not_parse_backend_provider_names() {
    use crate::arch::OpenAsrArchitectureRegistry;

    let root = models_root();
    let mut violations = Vec::new();
    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        let family_root = root.join(descriptor.identity.module_slug);
        let mut files = Vec::new();
        rust_files_below(&family_root, &mut files);
        for path in files {
            let syntax = ProductionSyntax::collect(&path);
            if syntax.parses_provider_names() {
                violations.push(format!(
                    "{} parses {:?}",
                    path.strip_prefix(&root).unwrap_or(&path).display(),
                    syntax.provider_name_parses
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "family policy must consume typed backend kinds/capabilities; raw provider-name parsing belongs in shared runtime code:\n{}",
        violations.join("\n")
    );
}

#[test]
fn native_backend_production_does_not_match_dolphin_architecture_directly() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/backend/native.rs");
    assert_production_does_not_reference(&path, "DOLPHIN_GGML_ARCHITECTURE_ID");
}

#[test]
fn qwen_shaped_families_quote_only_through_the_bound_decoder_contract() {
    let root = models_root();
    for (relative, required_call) in [
        (
            "funasr_nano/llm_transformer.rs",
            "quoted_qwen_decoder_system_memory_bytes",
        ),
        (
            "mimo_asr/llm_transformer.rs",
            "quoted_qwen_decoder_system_memory_bytes",
        ),
        (
            "firered_llm/llm_transformer.rs",
            "quoted_qwen_decoder_system_memory_bytes",
        ),
        (
            "moss_transcribe_diarize/prepared_runtime.rs",
            "add_qwen_decoder_prepared_runtime_quote",
        ),
    ] {
        let path = root.join(relative);
        let syntax = ProductionSyntax::collect(&path);
        assert!(
            syntax.calls_or_invokes_method(required_call),
            "{relative} must derive its decoder host quote from {required_call}"
        );
        for retired in [
            "quoted_retained_system_memory_bytes_for_family",
            "qwen_decoder_layer_tensor_descriptors",
            "qwen_decoder_tail_tensor_descriptors",
            "qwen_decoder_runtime_tensor_descriptors",
        ] {
            assert!(
                !syntax.references_identifier(retired),
                "{relative} must not reintroduce split Qwen decoder seam {retired}"
            );
        }
    }
}

#[test]
fn qwen_shaped_family_constructors_keep_the_bound_plan_tail_compile_chain() {
    let root = models_root();
    for (relative, binder) in [
        (
            "funasr_nano/llm_transformer.rs",
            "funasr_nano_qwen_decoder_contract",
        ),
        (
            "mimo_asr/llm_transformer.rs",
            "mimo_asr_qwen_decoder_contract",
        ),
        (
            "firered_llm/llm_transformer.rs",
            "firered_llm_qwen_decoder_contract",
        ),
    ] {
        let syntax = ProductionSyntax::collect(&root.join(relative));
        for required in [
            binder,
            "for_qwen_family",
            "load_qwen_decoder_tail_from_contract",
            "compile_qwen_whole_decoder_graph_from_prepared_plan",
        ] {
            assert!(
                syntax.calls_or_invokes_method(required),
                "{relative} must keep its production decoder on the bound contract chain; missing {required}"
            );
        }
    }

    let moss_prepare = "moss_transcribe_diarize/prepared_runtime.rs";
    let syntax = ProductionSyntax::collect(&root.join(moss_prepare));
    for required in [
        "moss_td_qwen_decoder_contract",
        "for_qwen_family",
        "load_qwen_decoder_tail_from_contract",
    ] {
        assert!(
            syntax.calls_or_invokes_method(required),
            "{moss_prepare} must keep its production decoder on the bound contract chain; missing {required}"
        );
    }
    let moss_compile = "moss_transcribe_diarize/llm_decoder.rs";
    assert!(
        ProductionSyntax::collect(&root.join(moss_compile))
            .calls_or_invokes_method("compile_qwen_whole_decoder_graph_from_prepared_plan"),
        "{moss_compile} must materialize the prepared decoder through the shared compile seam"
    );
}

#[test]
fn resident_model_actor_keys_exclude_request_capacity() {
    let root = models_root();
    for (relative, alias, expected) in [
        (
            "funasr_nano/executor.rs",
            "FunasrNanoDecoderRuntimeCacheKey",
            &["PackContentKey", "ExecutionLaneKey"][..],
        ),
        (
            "mimo_asr/executor.rs",
            "MimoAsrPreparedRuntimeCacheKey",
            &["PackContentKey", "ExecutionLaneKey"][..],
        ),
        (
            "firered_llm/executor.rs",
            "FireRedLlmDecoderCacheKey",
            &["PackContentKey", "ExecutionLaneKey"][..],
        ),
        (
            "moss_transcribe_diarize/executor.rs",
            "MossTdDecoderRuntimeCacheKey",
            &["PackContentKey", "ExecutionLaneKey"][..],
        ),
        (
            "granite_speech/executor.rs",
            "GraniteSpeechPreparedRuntimeCacheKey",
            &["PackContentKey", "ExecutionLaneKey"][..],
        ),
        (
            "qwen/ggml_executor.rs",
            "Qwen3AsrDecoderRuntimeCacheKey",
            &["PackContentKey", "ExecutionLaneKey", "String"][..],
        ),
    ] {
        assert_tuple_alias_components(&root.join(relative), alias, expected);
    }
}

#[test]
fn native_transcribe_production_does_not_match_whisper_architecture_directly() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/backend/native_transcribe.rs");
    assert_production_does_not_reference(&path, "WHISPER_GGML_ARCHITECTURE_ID");
}

#[test]
fn shared_decode_topologies_call_their_declared_driver() {
    use crate::arch::{OpenAsrArchitectureRegistry, OpenAsrDecodeDriverStrategy};

    let root = models_root();
    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        let (required_call, requires_truncation_forwarding) =
            match descriptor.topology_contract.decode_driver {
                OpenAsrDecodeDriverStrategy::SharedSeq2SeqGreedy { .. } => {
                    ("run_builtin_seq2seq_decode_policy", true)
                }
                OpenAsrDecodeDriverStrategy::SharedCtcGreedy { .. } => {
                    ("run_builtin_ctc_decode_policy", false)
                }
                OpenAsrDecodeDriverStrategy::Dedicated { .. } => continue,
            };

        let family_root = root.join(descriptor.identity.module_slug);
        let mut files = Vec::new();
        rust_files_below(&family_root, &mut files);
        let mut calls = BTreeSet::new();
        let mut methods = BTreeSet::new();
        for path in files {
            let syntax = ProductionSyntax::collect(&path);
            calls.extend(syntax.calls);
            methods.extend(syntax.methods);
        }

        assert!(
            calls.contains(required_call),
            "inventory family '{}' declares {:?} but its production AST never calls {required_call}",
            descriptor.identity.model_family,
            descriptor.topology_contract.decode_driver,
        );
        if requires_truncation_forwarding {
            assert!(
                methods.contains("into_decode_truncation"),
                "inventory family '{}' uses the shared seq2seq driver but never forwards its stop reason",
                descriptor.identity.model_family,
            );
        }
    }
}
