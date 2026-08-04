//! Force-linked pack-import surfaces for native families.
//!
//! Architecture descriptors carry the importer symbol and a compile-time
//! force-link callback. This module projects the builtin descriptor inventory
//! into the set consumed by runtime wiring, so a deleted or private convert
//! entry fails to compile at its descriptor and no independent symbol table can
//! drift from the architecture inventory.

use std::collections::BTreeSet;

use crate::arch::{OpenAsrArchitectureRegistry, OpenAsrPackImportSurface};

/// Returns the core convert symbols force-linked by the builtin architecture
/// inventory, keyed by symbol name.
pub(crate) fn linked_core_pack_import_symbols() -> BTreeSet<&'static str> {
    let mut linked = BTreeSet::new();
    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        if let OpenAsrPackImportSurface::CoreConvert { symbol, force_link } =
            descriptor.pack_contract.pack_import
        {
            force_link();
            assert!(
                linked.insert(symbol),
                "duplicate core pack-import symbol '{symbol}' in builtin architecture inventory"
            );
        }
    }
    linked
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn inventory_projection_covers_every_core_convert_surface() {
        let descriptors = OpenAsrArchitectureRegistry::with_builtins().descriptors();
        let linked = linked_core_pack_import_symbols();
        let mut inventory_symbols = BTreeSet::new();
        let mut core_convert_count = 0;

        for descriptor in descriptors {
            if let OpenAsrPackImportSurface::CoreConvert { symbol, force_link } =
                descriptor.pack_contract.pack_import
            {
                core_convert_count += 1;
                assert!(
                    inventory_symbols.insert(symbol),
                    "builtin architecture inventory must not repeat core pack-import symbol '{symbol}'"
                );
                force_link();
                assert!(
                    linked.contains(symbol),
                    "linked set must project '{}' from the descriptor inventory",
                    descriptor.identity.model_family
                );
            }
        }

        assert_eq!(
            linked.len(),
            core_convert_count,
            "linked set must contain one entry for every CoreConvert descriptor"
        );
        assert_eq!(
            linked, inventory_symbols,
            "linked set must contain only symbols declared by the descriptor inventory"
        );
    }
}
