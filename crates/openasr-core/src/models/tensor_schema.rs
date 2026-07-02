pub(crate) fn indexed_tensor_name(scope: &str, layer_idx: usize, slot: &str) -> String {
    format!("{scope}.{layer_idx}.{slot}")
}

/// Declares a per-layer tensor-name struct together with its builder from a
/// single data table mapping each struct field to its tensor-name leaf under a
/// shared block scope.
///
/// This is the LLM_TN-style name table for OpenASR: the `{scope}.{idx}.{slot}`
/// naming convention lives only in [`indexed_tensor_name`], so every
/// architecture contributes only data (field → leaf) and never re-implements
/// per-layer name formatting. The generated struct keeps named, compile-time
/// checked fields so consumers (`names.attn_q_weight`) and the binding specs
/// stay fully type-safe.
macro_rules! layer_tensor_names {
    (
        $(#[$struct_meta:meta])*
        $struct_vis:vis struct $struct_name:ident;
        $fn_vis:vis fn $builder:ident @ $scope:expr;
        {
            $( $field:ident => $slot:literal ),+ $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        $struct_vis struct $struct_name {
            $( pub $field: ::std::string::String, )+
        }

        $fn_vis fn $builder(layer_idx: usize) -> $struct_name {
            $struct_name {
                $(
                    $field: $crate::models::tensor_schema::indexed_tensor_name(
                        $scope, layer_idx, $slot,
                    ),
                )+
            }
        }
    };
}

pub(crate) use layer_tensor_names;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_tensor_name_builds_stable_dot_path() {
        assert_eq!(
            indexed_tensor_name("enc.blk", 5, "attn.q.weight"),
            "enc.blk.5.attn.q.weight"
        );
    }

    layer_tensor_names! {
        struct SampleLayerTensorNames;
        fn sample_layer_tensor_names @ "enc.blk";
        {
            attn_q_weight => "attn.q.weight",
            ffn_down_bias => "ffn_down.bias",
        }
    }

    #[test]
    fn layer_tensor_names_macro_resolves_scoped_paths() {
        let names = sample_layer_tensor_names(4);
        assert_eq!(names.attn_q_weight, "enc.blk.4.attn.q.weight");
        assert_eq!(names.ffn_down_bias, "enc.blk.4.ffn_down.bias");
    }
}
