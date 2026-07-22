use super::*;

#[test]
fn discovers_recognized_audio_and_video_files() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("sample.wav"));
    write_file(temp.path().join("meeting.mp4"));

    let discovered = discover_batch_inputs(temp.path()).unwrap();

    assert_eq!(
        file_names(&discovered.files),
        vec!["meeting.mp4", "sample.wav"]
    );
    assert_eq!(discovered.skipped_count, 0);
}

#[test]
fn discovery_is_case_insensitive() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("SAMPLE.WAV"));
    write_file(temp.path().join("clip.MP3"));

    let discovered = discover_batch_inputs(temp.path()).unwrap();

    assert_eq!(
        file_names(&discovered.files),
        vec!["SAMPLE.WAV", "clip.MP3"]
    );
}

#[test]
fn discovery_skips_unsupported_extensions() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("sample.wav"));
    write_file(temp.path().join("notes.txt"));

    let discovered = discover_batch_inputs(temp.path()).unwrap();

    assert_eq!(file_names(&discovered.files), vec!["sample.wav"]);
    assert_eq!(discovered.skipped_count, 1);
}

#[test]
fn discovery_skips_directories() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("sample.wav"));
    fs::create_dir(temp.path().join("nested.mp3")).unwrap();

    let discovered = discover_batch_inputs(temp.path()).unwrap();

    assert_eq!(file_names(&discovered.files), vec!["sample.wav"]);
    assert_eq!(discovered.skipped_count, 1);
}

#[test]
fn discovery_ordering_is_deterministic() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("zeta.wav"));
    write_file(temp.path().join("alpha.wav"));
    write_file(temp.path().join("middle.mp3"));

    let discovered = discover_batch_inputs(temp.path()).unwrap();

    assert_eq!(
        file_names(&discovered.files),
        vec!["alpha.wav", "middle.mp3", "zeta.wav"]
    );
}

#[test]
fn missing_input_directory_returns_friendly_error() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");

    let error = discover_batch_inputs(&missing).unwrap_err().to_string();

    assert!(error.contains("Batch input directory not found:"));
    assert!(error.contains(
        "Please provide an existing directory containing supported audio or video files."
    ));
}

#[test]
fn file_input_returns_friendly_not_directory_error() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.wav");
    write_file(&input);

    let error = discover_batch_inputs(&input).unwrap_err().to_string();

    assert!(error.contains("Batch input path is not a directory:"));
    assert!(error.contains(
        "Please provide an existing directory containing supported audio or video files."
    ));
}

#[test]
fn no_supported_files_returns_friendly_error_with_supported_extensions() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path().join("notes.txt"));

    let error = discover_batch_inputs(temp.path()).unwrap_err().to_string();

    assert!(error.contains("No supported audio or video files found in:"));
    assert!(
        error.contains("Supported extensions: wav, mp3, mp4, m4a, webm, flac, ogg, opus, qta.")
    );
}

#[test]
fn output_extension_mapping_is_stable() {
    assert_eq!(response_format_extension(ResponseFormat::Text), "txt");
    assert_eq!(response_format_extension(ResponseFormat::Json), "json");
    assert_eq!(
        response_format_extension(ResponseFormat::VerboseJson),
        "verbose.json"
    );
    assert_eq!(response_format_extension(ResponseFormat::Srt), "srt");
    assert_eq!(response_format_extension(ResponseFormat::Vtt), "vtt");
    assert_eq!(response_format_extension(ResponseFormat::Markdown), "md");
}

#[test]
fn output_path_includes_original_file_name_to_avoid_stem_collisions() {
    let output_dir = PathBuf::from("/tmp/openasr-batch");

    assert_eq!(
        batch_output_path(&output_dir, "sample.wav", ResponseFormat::Text),
        output_dir.join("sample.wav.txt")
    );
    assert_eq!(
        batch_output_path(&output_dir, "sample.mp3", ResponseFormat::Text),
        output_dir.join("sample.mp3.txt")
    );
    assert_eq!(
        batch_output_path(&output_dir, "meeting.wav", ResponseFormat::VerboseJson),
        output_dir.join("meeting.wav.verbose.json")
    );
}

#[test]
fn summary_rendering_is_stable_and_omits_transcript_text() {
    let summary = BatchSummary {
        input_dir: PathBuf::from("fixtures"),
        output_dir: PathBuf::from("/tmp/openasr-batch"),
        format: ResponseFormat::Srt,
        model: "whisper-tiny".to_string(),
        backend: "mock".to_string(),
        files_found: 2,
        files_transcribed: 2,
        files_skipped: 1,
        files_failed: 0,
        outputs: vec![BatchOutput {
            input_path: PathBuf::from("fixtures/jfk.wav"),
            output_path: PathBuf::from("/tmp/openasr-batch/sample.wav.srt"),
        }],
        failures: Vec::new(),
    };

    let rendered = render_batch_summary(&summary);

    assert_eq!(
        rendered,
        "OpenASR batch transcription\n\nInput directory: fixtures\nOutput directory: /tmp/openasr-batch\nFormat: srt\nModel: whisper-tiny\nBackend: mock\nFiles found: 2\nFiles transcribed: 2\nFiles skipped: 1\nFiles failed: 0\n\nOutputs:\n- fixtures/jfk.wav -> /tmp/openasr-batch/sample.wav.srt\n"
    );
    assert!(!rendered.contains("OpenASR mock transcription"));
}

#[test]
fn summary_renders_concise_failure_entries() {
    let summary = BatchSummary {
        input_dir: PathBuf::from("fixtures"),
        output_dir: PathBuf::from("/tmp/openasr-batch"),
        format: ResponseFormat::Text,
        model: "whisper-tiny".to_string(),
        backend: "native".to_string(),
        files_found: 2,
        files_transcribed: 1,
        files_skipped: 0,
        files_failed: 1,
        outputs: vec![BatchOutput {
            input_path: PathBuf::from("fixtures/good.wav"),
            output_path: PathBuf::from("/tmp/openasr-batch/good.wav.txt"),
        }],
        failures: vec![BatchFailure {
            input_path: PathBuf::from("fixtures/bad.mp3"),
            error: "Could not convert input audio.\nDetailed help line.".to_string(),
        }],
    };

    let rendered = render_batch_summary(&summary);

    assert!(rendered.contains("Files failed: 1"));
    assert!(rendered.contains("Failures:"));
    assert!(rendered.contains("fixtures/bad.mp3: Could not convert input audio."));
    assert!(!rendered.contains("Detailed help line."));
}

fn write_file(path: impl AsRef<Path>) {
    fs::write(path, b"audio").unwrap();
}

fn file_names(files: &[BatchItem]) -> Vec<String> {
    files
        .iter()
        .map(|item| {
            item.input_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}
