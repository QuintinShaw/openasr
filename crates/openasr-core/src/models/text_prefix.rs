/// Byte length of the longest shared leading run of `a` and `b`, aligned to a
/// `char` boundary so slicing either string at the returned offset is valid.
pub(crate) fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut offset = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        offset += ca.len_utf8();
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_shared_ascii_byte_len() {
        assert_eq!(common_prefix_len("openasr-core", "openasr-server"), 8);
    }

    #[test]
    fn returns_utf8_boundary_len() {
        let left = "hello \u{4f60}\u{597d} world";
        let right = "hello \u{4f60}\u{597d} rust";

        assert_eq!(
            common_prefix_len(left, right),
            "hello \u{4f60}\u{597d} ".len()
        );
    }

    #[test]
    fn stops_before_first_mismatch() {
        assert_eq!(common_prefix_len("abc", "xbc"), 0);
        assert_eq!(common_prefix_len("abc", "ab"), 2);
    }
}
