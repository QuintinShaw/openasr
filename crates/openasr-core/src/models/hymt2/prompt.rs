use super::tokenizer::{
    HYMT2_ASSISTANT_TOKEN_ID, HYMT2_BOS_TOKEN_ID, HYMT2_USER_TOKEN_ID, Hymt2Tokenizer,
};
use crate::NativeAsrError;

/// Hunyuan-MT is a translation-specialized fine-tune, not a general
/// instruction follower: it is trained on this exact Chinese zh→en prompt
/// shape ("把下面的文本翻译成英文，不要额外解释。"). Deviating from it —
/// especially with an English meta-prompt plus a labeled Context section of
/// `source = target` pairs — measurably breaks it: the model starts
/// translating the context lines instead of the source and echoes the
/// ` = ` separator into its output (seen verbatim in production captions).
pub const HYMT2_SUBTITLE_TRANSLATION_INSTRUCTION: &str = "把下面的文本翻译成英文，不要额外解释。";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hymt2SubtitlePromptTokenParts {
    pub prompt_tokens: Vec<u32>,
    pub source_prefix_tokens: Vec<u32>,
    pub static_context_token_count: usize,
    pub generation_marker_tokens: Vec<u32>,
    pub source_tokens: Vec<u32>,
}

pub fn build_subtitle_translation_prompt(
    source_clause: &str,
    finalized_context: &[(&str, &str)],
) -> String {
    let mut prompt = build_subtitle_translation_prompt_prefix(finalized_context);
    prompt.push_str(source_clause.trim());
    prompt
}

pub(crate) fn build_subtitle_translation_prompt_prefix(
    finalized_context: &[(&str, &str)],
) -> String {
    // No context section: feeding labeled `source = target` history confuses
    // the translation-specialized model (see the instruction doc above). The
    // constant prefix also lets the prefix cache survive across clauses
    // instead of resetting whenever the rolling context window advances.
    let _ = finalized_context;
    let mut prompt = String::with_capacity(64);
    prompt.push_str(HYMT2_SUBTITLE_TRANSLATION_INSTRUCTION);
    prompt.push_str("\n\n");
    prompt
}

pub fn build_hymt2_user_chat_prompt_tokens(
    tokenizer: &Hymt2Tokenizer,
    user_content: &str,
) -> Result<Vec<u32>, NativeAsrError> {
    tokenizer.encode_user_chat_prompt(user_content)
}

pub(crate) fn build_hymt2_subtitle_prompt_token_parts(
    tokenizer: &Hymt2Tokenizer,
    source_clause: &str,
    finalized_context: &[(&str, &str)],
) -> Result<Hymt2SubtitlePromptTokenParts, NativeAsrError> {
    let prefix_text = build_subtitle_translation_prompt_prefix(finalized_context);
    let source_text = source_clause.trim();
    let prompt_text = {
        let mut text = String::with_capacity(prefix_text.len().saturating_add(source_text.len()));
        text.push_str(&prefix_text);
        text.push_str(source_text);
        text
    };
    let prompt_tokens = tokenizer.encode_user_chat_prompt(&prompt_text)?;
    let prefix_content_tokens = tokenizer.encode_content_text(&prefix_text)?;
    let source_tokens = tokenizer.encode_content_text(source_text)?;

    let mut static_context_tokens = Vec::with_capacity(prefix_content_tokens.len() + 2);
    static_context_tokens.push(HYMT2_BOS_TOKEN_ID);
    static_context_tokens.push(HYMT2_USER_TOKEN_ID);
    static_context_tokens.extend(prefix_content_tokens);

    let mut source_prefix_tokens = Vec::with_capacity(
        static_context_tokens
            .len()
            .saturating_add(source_tokens.len()),
    );
    source_prefix_tokens.extend_from_slice(&static_context_tokens);
    source_prefix_tokens.extend_from_slice(&source_tokens);
    let generation_marker_tokens = vec![HYMT2_ASSISTANT_TOKEN_ID];

    let mut recomposed_prompt = source_prefix_tokens.clone();
    recomposed_prompt.extend_from_slice(&generation_marker_tokens);
    if recomposed_prompt != prompt_tokens {
        let source_prefix_tokens = prompt_tokens
            .strip_suffix(generation_marker_tokens.as_slice())
            .unwrap_or(&prompt_tokens)
            .to_vec();
        return Ok(Hymt2SubtitlePromptTokenParts {
            prompt_tokens,
            static_context_token_count: 0,
            source_prefix_tokens,
            generation_marker_tokens,
            source_tokens,
        });
    }

    Ok(Hymt2SubtitlePromptTokenParts {
        prompt_tokens,
        static_context_token_count: static_context_tokens.len(),
        source_prefix_tokens,
        generation_marker_tokens,
        source_tokens,
    })
}

pub fn max_output_tokens_for_source_tokens(source_tokens: usize) -> usize {
    96.min(24.max(source_tokens.saturating_mul(2).saturating_add(16)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_output_tokens_matches_mvp_formula() {
        assert_eq!(max_output_tokens_for_source_tokens(0), 24);
        assert_eq!(max_output_tokens_for_source_tokens(4), 24);
        assert_eq!(max_output_tokens_for_source_tokens(20), 56);
        assert_eq!(max_output_tokens_for_source_tokens(100), 96);
    }

    #[test]
    fn prompt_uses_stable_user_only_content_without_system_role() {
        let prompt = build_subtitle_translation_prompt("我们需要保持流式路径很快。", &[]);
        assert!(prompt.starts_with(HYMT2_SUBTITLE_TRANSLATION_INSTRUCTION));
        assert!(prompt.ends_with("\n\n我们需要保持流式路径很快。"));
        assert!(!prompt.contains("System:"));
    }

    #[test]
    fn prompt_prefix_stays_constant_regardless_of_finalized_context() {
        // The translation-specialized model must never see a labeled context
        // section (it starts translating the context instead of the source),
        // and a constant prefix is what lets the prefix cache survive across
        // clauses as the rolling context window advances.
        let without_context = build_subtitle_translation_prompt_prefix(&[]);
        let with_context =
            build_subtitle_translation_prompt_prefix(&[("源文", "translation"), ("再来", "again")]);
        assert_eq!(without_context, with_context);
        assert!(!with_context.contains('='));
        assert!(!with_context.contains("Context"));
    }
}
