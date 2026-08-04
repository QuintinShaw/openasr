//! Source-level architecture gate for the GGUF runtime provenance boundary.
//!
//! Rust's type system enforces preflight once a constructor accepts the typed
//! value. This audit closes the remaining escape hatches: a new family must
//! not call an unpreflighted tensor-reader or bare-source native weight loader
//! from production runtime code. Import/conversion boundaries and test-only
//! modules deliberately remain able to inspect arbitrary GGUF sources.

use std::fs;
use std::path::{Path, PathBuf};

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprMethodCall, ExprPath, ExprStruct, File, ImplItemFn, ItemFn, ItemImpl,
    ItemMod, Lit, Member, Meta, Token,
};

const UNPREFLIGHTED_READER_METHODS: &[&str] =
    &["from_path", "from_runtime_source", "from_preflight_parts"];
const DIRECT_METADATA_FUNCTIONS: &[&str] = &[
    "read_gguf_metadata",
    "read_gguf_metadata_from_runtime_source",
    "read_gguf_metadata_from_runtime_source_with_limits",
    "read_gguf_metadata_from_context",
    "read_gguf_tensor_index",
    "read_gguf_tensor_index_from_runtime_source",
    "read_gguf_tensor_index_from_runtime_source_with_limits",
    "read_gguf_tensor_index_from_context",
    "load_gguf_metadata_and_tensor_index_with_c_parser_sandbox",
];

#[derive(Debug)]
struct Violation {
    path: PathBuf,
    operation: String,
}

struct ProductionRuntimeVisitor<'a> {
    path: &'a Path,
    violations: Vec<Violation>,
}

impl ProductionRuntimeVisitor<'_> {
    fn reject(&mut self, operation: impl Into<String>) {
        self.violations.push(Violation {
            path: self.path.to_path_buf(),
            operation: operation.into(),
        });
    }
}

impl<'ast> Visit<'ast> for ProductionRuntimeVisitor<'_> {
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
            visit::visit_item_impl(self, node);
        }
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if !is_test_only(&node.attrs) {
            visit::visit_impl_item_fn(self, node);
        }
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if node.method == "load_gguf_weight_context" {
            self.reject("bare-source load_gguf_weight_context");
        }
        if node.method == "ok"
            && let Expr::Call(call) = node.receiver.as_ref()
            && let Expr::Path(function) = call.func.as_ref()
            && function
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "load_gguf_weight_context_from_preflight")
        {
            self.reject("suppressed load_gguf_weight_context_from_preflight error");
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_path(&mut self, node: &'ast ExprPath) {
        let segments: Vec<_> = node
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        if segments.len() >= 2
            && segments[segments.len() - 2] == "GgufTensorDataReader"
            && UNPREFLIGHTED_READER_METHODS
                .iter()
                .any(|method| segments.last().is_some_and(|last| last == method))
        {
            self.reject(format!(
                "unpreflighted tensor reader {}",
                segments.last().expect("non-empty path")
            ));
        }
        if let Some(last) = segments.last()
            && DIRECT_METADATA_FUNCTIONS
                .iter()
                .any(|function| last == function)
        {
            self.reject(format!("direct GGUF header read {last}"));
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_expr_struct(&mut self, node: &'ast ExprStruct) {
        for field in &node.fields {
            let Member::Named(member) = &field.member else {
                continue;
            };
            if member == "runtime_source_preflight"
                && matches!(
                    &field.expr,
                    syn::Expr::Path(path)
                        if path.path.segments.last().is_some_and(|segment| segment.ident == "None")
                )
            {
                self.reject("production request constructed without runtime preflight");
            }
        }
        visit::visit_expr_struct(self, node);
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
                    syn::Expr::Lit(literal)
                        if matches!(&literal.lit, Lit::Str(feature) if feature.value() == "testing")
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

fn is_model_package_import_boundary(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) != Some("package_import.rs") {
        return false;
    }
    let Some(parent) = path.parent() else {
        return false;
    };

    // Every model family owns its import/conversion seam in a package_import.rs
    // module. Match that shape instead of requiring a central family list.
    path.ancestors()
        .skip(1)
        .any(|ancestor| ancestor.ends_with(Path::new("src/models")) && ancestor != parent)
}

fn is_explicit_import_boundary(path: &Path) -> bool {
    const IMPORT_BOUNDARIES: &[&str] = &[
        // Non-standard import/conversion seams that are not named
        // `src/models/<family>/package_import.rs`.
        "src/models/diarize_pack_import.rs",
        "src/models/local_source_import.rs",
        "src/models/qwen/forced_aligner_import.rs",
        // Standalone public inspection ingress: it has no existing execution
        // preflight to reuse and returns only model identity.
        "src/api/backend/native_model_id.rs",
    ];
    is_model_package_import_boundary(path)
        || IMPORT_BOUNDARIES
            .iter()
            .any(|boundary| path.ends_with(boundary))
}

fn is_test_source(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "tests")
        || path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| {
                stem == "tests" || stem.ends_with("_tests") || stem.ends_with("_bench")
            })
}

fn collect_rust_sources(root: &Path, output: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", root.display()))
        .map(|entry| entry.expect("source directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

fn audit_file(path: &Path) -> Vec<Violation> {
    if is_explicit_import_boundary(path) || is_test_source(path) {
        return Vec::new();
    }
    let source = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    let syntax: File = syn::parse_file(&source)
        .unwrap_or_else(|error| panic!("could not parse {}: {error}", path.display()));
    let mut visitor = ProductionRuntimeVisitor {
        path,
        violations: Vec::new(),
    };
    visitor.visit_file(&syntax);
    visitor.violations
}

#[test]
fn production_runtime_construction_cannot_bypass_gguf_preflight() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        crate_root.join("src/models"),
        crate_root.join("src/diarize"),
        crate_root.join("src/api/backend"),
    ];
    let mut sources = Vec::new();
    for root in roots {
        collect_rust_sources(&root, &mut sources);
    }

    let violations: Vec<_> = sources.iter().flat_map(|path| audit_file(path)).collect();
    assert!(
        violations.is_empty(),
        "production GGUF runtime construction bypassed GgufRuntimeSourcePreflight:\n{}",
        violations
            .iter()
            .map(|violation| format!("- {}: {}", violation.path.display(), violation.operation))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_model_family_package_import_is_an_import_boundary() {
        assert!(is_model_package_import_boundary(Path::new(
            "crates/openasr-core/src/models/new_family/package_import.rs"
        )));
        assert!(is_model_package_import_boundary(Path::new(
            "src/models/new_family/variant/package_import.rs"
        )));
    }

    #[test]
    fn import_boundary_shape_is_fail_closed() {
        assert!(!is_model_package_import_boundary(Path::new(
            "crates/openasr-core/src/models/package_import.rs"
        )));
        assert!(!is_model_package_import_boundary(Path::new(
            "crates/openasr-core/src/models/new_family/import.rs"
        )));
        assert!(!is_model_package_import_boundary(Path::new(
            "crates/openasr-core/src/model_sources/new_family/package_import.rs"
        )));
    }

    #[test]
    fn non_standard_import_boundaries_remain_explicit_exceptions() {
        assert!(is_explicit_import_boundary(Path::new(
            "src/models/diarize_pack_import.rs"
        )));
        assert!(is_explicit_import_boundary(Path::new(
            "src/models/local_source_import.rs"
        )));
        assert!(is_explicit_import_boundary(Path::new(
            "src/models/qwen/forced_aligner_import.rs"
        )));
        assert!(is_explicit_import_boundary(Path::new(
            "src/api/backend/native_model_id.rs"
        )));
    }
}
