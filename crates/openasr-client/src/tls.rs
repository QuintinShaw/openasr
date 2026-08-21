use std::{io, sync::Arc, time::Duration};

use rustls::pki_types::ServerName;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};
use tokio_rustls::TlsConnector;

use crate::{ClientError, TofuServerVerifier, tls_client_config};

pub(crate) const REMOTE_COMPUTE_HEADER: &str = "X-OpenASR-Remote-Compute";
pub(crate) const REMOTE_COMPUTE_CLIENT_VALUE: &str = "client";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const IO_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct RemoteHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub server_fingerprint: String,
}

/// Opens the TOFU-pinned TLS connection shared by HTTP and WebSocket.
///
/// Hostname/SAN matching is not a hard failure: [`TofuServerVerifier`] authenticates
/// the peer by certificate fingerprint (and a real signature proving key
/// possession). That is what makes LAN IP / `0.0.0.0` binds usable when the
/// self-signed cert only names `127.0.0.1` or `localhost`.
pub(crate) async fn open_remote_tls_connection(
    host: &str,
    port: u16,
    expected_fingerprint: Option<&str>,
) -> Result<(tokio_rustls::client::TlsStream<TcpStream>, String), ClientError> {
    let verifier = Arc::new(TofuServerVerifier::new(
        expected_fingerprint.map(str::to_string),
    ));
    let tls_config = tls_client_config(verifier.clone()).map_err(ClientError::new)?;
    let connector = TlsConnector::from(tls_config);
    let server_name = ServerName::try_from(host.to_string()).map_err(|_| {
        ClientError::new("Remote server host is not a valid DNS name or IP address.")
    })?;
    let tcp = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect((host, port)))
        .await
        .map_err(|_| ClientError::new("Timed out connecting to the OpenASR remote server."))?
        .map_err(|error| {
            ClientError::new(format!(
                "Could not connect to the OpenASR remote server: {error}"
            ))
        })?;
    let tls = tokio::time::timeout(CONNECT_TIMEOUT, connector.connect(server_name, tcp))
        .await
        .map_err(|_| ClientError::new("Timed out during OpenASR remote TLS handshake."))?
        .map_err(|error| {
            ClientError::new(format!("OpenASR remote TLS handshake failed: {error}"))
        })?;
    let server_fingerprint = verifier.fingerprint().ok_or_else(|| {
        ClientError::new("OpenASR remote server did not present a TLS certificate.")
    })?;
    Ok((tls, server_fingerprint))
}

pub(crate) fn host_header(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn header_value_is_safe(value: &str) -> bool {
    !value.contains('\r') && !value.contains('\n')
}

#[allow(clippy::too_many_arguments)]
fn build_remote_request_header(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    accept: &str,
    content_length: u64,
    content_type: Option<&str>,
    bearer_token: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> Result<Vec<u8>, ClientError> {
    let host_value = host_header(host, port);
    if !header_value_is_safe(&host_value)
        || !header_value_is_safe(path)
        || !header_value_is_safe(method)
        || bearer_token.is_some_and(|token| !header_value_is_safe(token))
    {
        return Err(ClientError::new(
            "OpenASR remote request contains invalid header characters.",
        ));
    }
    let mut request_header = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_value}\r\nAccept: {accept}\r\nContent-Length: {content_length}\r\nConnection: close\r\n"
    );
    if let Some(content_type) = content_type {
        request_header.push_str("Content-Type: ");
        request_header.push_str(content_type);
        request_header.push_str("\r\n");
    }
    if let Some(token) = bearer_token {
        request_header.push_str("Authorization: Bearer ");
        request_header.push_str(token);
        request_header.push_str("\r\n");
    }
    for (name, value) in extra_headers {
        if header_value_is_safe(name) && header_value_is_safe(value) {
            request_header.push_str(name);
            request_header.push_str(": ");
            request_header.push_str(value);
            request_header.push_str("\r\n");
        }
    }
    request_header.push_str("\r\n");
    Ok(request_header.into_bytes())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_remote_http_request(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Vec<u8>,
    content_type: Option<&str>,
    accept: &str,
    expected_fingerprint: Option<&str>,
    bearer_token: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> Result<RemoteHttpResponse, ClientError> {
    let (mut tls, server_fingerprint) =
        open_remote_tls_connection(host, port, expected_fingerprint).await?;
    let header = build_remote_request_header(
        host,
        port,
        method,
        path,
        accept,
        body.len() as u64,
        content_type,
        bearer_token,
        extra_headers,
    )?;
    tokio::time::timeout(CONNECT_TIMEOUT, tls.write_all(&header))
        .await
        .map_err(|_| ClientError::new("Timed out sending OpenASR remote request."))?
        .map_err(|error| {
            ClientError::new(format!("Could not send OpenASR remote request: {error}"))
        })?;
    tokio::time::timeout(CONNECT_TIMEOUT, tls.write_all(&body))
        .await
        .map_err(|_| ClientError::new("Timed out sending OpenASR remote request."))?
        .map_err(|error| {
            ClientError::new(format!("Could not send OpenASR remote request: {error}"))
        })?;
    tokio::time::timeout(CONNECT_TIMEOUT, tls.flush())
        .await
        .map_err(|_| ClientError::new("Timed out flushing OpenASR remote request."))?
        .map_err(|error| {
            ClientError::new(format!("Could not flush OpenASR remote request: {error}"))
        })?;
    let response = read_tls_http_response(&mut tls).await?;
    let parsed = parse_http_response(&response)?;
    Ok(RemoteHttpResponse {
        status: parsed.status,
        headers: parsed.headers,
        body: parsed.body.to_vec(),
        server_fingerprint,
    })
}

async fn read_tls_http_response(
    tls: &mut tokio_rustls::client::TlsStream<TcpStream>,
) -> Result<Vec<u8>, ClientError> {
    let mut response = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        match tokio::time::timeout(IO_TIMEOUT, tls.read(&mut buffer)).await {
            Err(_) => {
                return Err(ClientError::new(
                    "Timed out waiting for OpenASR remote response.",
                ));
            }
            Ok(Ok(0)) => return Ok(response),
            Ok(Ok(read)) => {
                response.extend_from_slice(&buffer[..read]);
                if response.len() > MAX_RESPONSE_BYTES {
                    return Err(ClientError::new(
                        "OpenASR remote response exceeded the maximum allowed size.",
                    ));
                }
            }
            Ok(Err(error))
                if error.kind() == io::ErrorKind::UnexpectedEof && !response.is_empty() =>
            {
                return Ok(response);
            }
            Ok(Err(error)) => {
                return Err(ClientError::new(format!(
                    "Could not read OpenASR remote response: {error}"
                )));
            }
        }
    }
}

struct ParsedHttpResponse<'a> {
    status: u16,
    headers: Vec<(String, String)>,
    body: &'a [u8],
}

fn parse_http_response(response: &[u8]) -> Result<ParsedHttpResponse<'_>, ClientError> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| ClientError::new("OpenASR remote response was not valid HTTP."))?;
    let header = std::str::from_utf8(&response[..header_end])
        .map_err(|_| ClientError::new("OpenASR remote response header was not UTF-8."))?;
    let mut lines = header.lines();
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| ClientError::new("OpenASR remote response did not include a status."))?;
    let headers = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_string(), value.trim().to_string()))
        })
        .collect::<Vec<(String, String)>>();
    let mut body = &response[header_end + 4..];
    if let Some((_, length)) = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        && let Ok(length) = length.parse::<usize>()
    {
        body = &body[..body.len().min(length)];
    }
    Ok(ParsedHttpResponse {
        status,
        headers,
        body,
    })
}

pub(crate) fn content_type_of(headers: &[(String, String)]) -> String {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_header_brackets_ipv6() {
        assert_eq!(host_header("::1", 8443), "[::1]:8443");
        assert_eq!(host_header("127.0.0.1", 8080), "127.0.0.1:8080");
    }

    #[test]
    fn parses_http_pairing_response_status_and_body() {
        let response = b"HTTP/1.1 202 Accepted\r\nContent-Length: 2\r\n\r\n{}";
        let parsed = parse_http_response(response).unwrap();
        assert_eq!(parsed.status, 202);
        assert_eq!(parsed.body, b"{}");
    }
}
