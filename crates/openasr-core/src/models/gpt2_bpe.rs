use std::collections::BTreeMap;

use crate::NativeAsrError;

pub(crate) fn build_merge_rank(merges: &[String]) -> BTreeMap<String, usize> {
    merges
        .iter()
        .enumerate()
        .map(|(index, merge)| (merge.clone(), index))
        .collect()
}

pub(crate) fn build_token_to_id(
    tokens: &[String],
    tokenizer_name: &str,
) -> Result<BTreeMap<String, u32>, NativeAsrError> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let token_id =
                u32::try_from(index).map_err(|_| NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "{tokenizer_name} tokenizer token index {index} exceeds u32 range"
                    ),
                })?;
            Ok((token.clone(), token_id))
        })
        .collect()
}

pub(crate) fn token_to_bytes(token: &str) -> Vec<u8> {
    token
        .chars()
        .map(|ch| byte_from_unicode_char(ch).unwrap_or(b'?'))
        .collect()
}

pub(crate) fn bytes_to_unicode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| unicode_char_from_byte(*byte))
        .collect()
}

pub(crate) fn encode_prompt_text(
    text: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let mut token_ids = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        if let Some((token_id, next_cursor)) = try_match_special_token(text, cursor, token_to_id) {
            token_ids.push(token_id);
            cursor = next_cursor;
            continue;
        }

        let next_special = find_next_special_token_boundary(text, cursor, token_to_id);
        let chunk = &text[cursor..next_special];
        token_ids.extend(encode_plain_text_chunk(
            chunk,
            token_to_id,
            merge_rank,
            tokenizer_name,
        )?);
        cursor = next_special;
    }
    Ok(token_ids)
}

fn try_match_special_token(
    text: &str,
    cursor: usize,
    token_to_id: &BTreeMap<String, u32>,
) -> Option<(u32, usize)> {
    let rest = text.get(cursor..)?;
    if !rest.starts_with("<|") {
        return None;
    }
    let end_offset = rest.find("|>")?;
    let token = &rest[..end_offset + 2];
    let token_id = *token_to_id.get(token)?;
    Some((token_id, cursor + end_offset + 2))
}

fn find_next_special_token_boundary(
    text: &str,
    start: usize,
    token_to_id: &BTreeMap<String, u32>,
) -> usize {
    let mut cursor = start;
    while cursor < text.len() {
        let Some(relative) = text[cursor..].find("<|") else {
            return text.len();
        };
        let candidate = cursor + relative;
        if try_match_special_token(text, candidate, token_to_id).is_some() {
            return candidate;
        }
        cursor = candidate.saturating_add(1);
    }
    text.len()
}

fn encode_plain_text_chunk(
    chunk: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let mut token_ids = Vec::new();
    let mut cursor = 0usize;
    while cursor < chunk.len() {
        let start = cursor;
        let Some(first_char) = chunk[cursor..].chars().next() else {
            break;
        };
        if first_char.is_whitespace() {
            cursor += first_char.len_utf8();
        }
        while cursor < chunk.len() {
            let Some(ch) = chunk[cursor..].chars().next() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            cursor += ch.len_utf8();
        }
        if cursor == start {
            cursor += first_char.len_utf8();
        }
        let piece = &chunk[start..cursor];
        if piece.is_empty() {
            continue;
        }
        token_ids.extend(encode_byte_level_piece(
            piece,
            token_to_id,
            merge_rank,
            tokenizer_name,
        )?);
    }
    Ok(token_ids)
}

pub(crate) fn encode_byte_level_piece(
    piece: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    let encoded = bytes_to_unicode(piece.as_bytes());
    bpe_encode_piece(&encoded, token_to_id, merge_rank, tokenizer_name)
}

fn bpe_encode_piece(
    encoded_piece: &str,
    token_to_id: &BTreeMap<String, u32>,
    merge_rank: &BTreeMap<String, usize>,
    tokenizer_name: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    if encoded_piece.is_empty() {
        return Ok(Vec::new());
    }
    let mut symbols = encoded_piece
        .chars()
        .map(|ch| ch.to_string())
        .collect::<Vec<_>>();
    while symbols.len() > 1 {
        let mut best_pair: Option<(usize, usize)> = None;
        for pair_index in 0..symbols.len() - 1 {
            let key = format!("{} {}", symbols[pair_index], symbols[pair_index + 1]);
            let Some(rank) = merge_rank.get(&key).copied() else {
                continue;
            };
            match best_pair {
                Some((_, best_rank)) if rank >= best_rank => {}
                _ => best_pair = Some((pair_index, rank)),
            }
        }
        let Some((pair_index, _)) = best_pair else {
            break;
        };
        let merged = format!("{}{}", symbols[pair_index], symbols[pair_index + 1]);
        symbols.splice(pair_index..=pair_index + 1, [merged]);
    }

    symbols
        .into_iter()
        .map(|symbol| {
            token_to_id.get(&symbol).copied().ok_or_else(|| {
                NativeAsrError::UnsupportedModelPack {
                    reason: format!(
                        "{tokenizer_name} tokenizer cannot encode BPE piece '{symbol}' from prompt text"
                    ),
                }
            })
        })
        .collect()
}

fn byte_from_unicode_char(ch: char) -> Option<u8> {
    let code = u32::from(ch);
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return u8::try_from(code).ok();
    }
    if code < 256 {
        return None;
    }

    let mut extra_code = 256u32;
    for byte in 0u32..=255 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
        {
            continue;
        }
        if extra_code == code {
            return u8::try_from(byte).ok();
        }
        extra_code += 1;
    }
    None
}

fn unicode_char_from_byte(byte: u8) -> char {
    let code = u32::from(byte);
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return char::from(byte);
    }

    let mut extra_code = 256u32;
    for candidate in 0u32..=255 {
        if (33..=126).contains(&candidate)
            || (161..=172).contains(&candidate)
            || (174..=255).contains(&candidate)
        {
            continue;
        }
        if candidate == code {
            return char::from_u32(extra_code).unwrap_or('?');
        }
        extra_code += 1;
    }
    '?'
}
