#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = openasr_server::fuzz::fuzz_parse_client_message(data);
});
