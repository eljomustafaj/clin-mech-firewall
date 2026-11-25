use std::io::{BufRead, Write};

use serde::{Deserialize, Deserializer, Serialize};

use crate::policy::{Decision, Policy};
use crate::Result;

// Integers on the wire must retain their exact value in the browser.
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    schema_version: u64,
    trace_id: String,
    model_id: String,
    model_revision: String,
    layer: u64,
    probe_id: String,
    probe_revision: String,
}

impl Source {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err("unsupported schema version".into());
        }
        if self.layer > MAX_SAFE_INTEGER {
            return Err("layer exceeds the safe integer range".into());
        }
        for id in [
            &self.trace_id,
            &self.model_id,
            &self.model_revision,
            &self.probe_id,
            &self.probe_revision,
        ] {
            // Python's str.isspace() also includes the four information separators.
            if id
                .chars()
                .all(|c| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            {
                return Err("source identifiers and revisions must contain text".into());
            }
        }
        Ok(())
    }
}

// An absent feature list is valid. An explicitly supplied list must be an array.
fn feature_ids<'de, D: Deserializer<'de>>(
    input: D,
) -> std::result::Result<Option<Vec<u64>>, D::Error> {
    Vec::deserialize(input).map(Some)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Token {
    index: u64,
    text: String,
    risk: f64,
    #[serde(
        default,
        deserialize_with = "feature_ids",
        skip_serializing_if = "Option::is_none"
    )]
    sae_feature_ids: Option<Vec<u64>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum Input {
    Start(Source),
    Token(Token),
    End { token_count: u64 },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Event<'a> {
    Start {
        #[serde(flatten)]
        source: &'a Source,
        threshold: f64,
    },
    Decision {
        trace_id: &'a str,
        #[serde(flatten)]
        token: &'a Token,
        decision: Decision,
    },
    End {
        trace_id: &'a str,
        status: &'static str,
        token_count: u64,
    },
}

fn emit(out: &mut impl Write, event: Event<'_>) -> Result<()> {
    serde_json::to_writer(&mut *out, &event)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn parse(line: Option<std::io::Result<String>>, number: u64) -> Result<Input> {
    let line = line
        .ok_or_else(|| format!("unexpected EOF before line {number}"))?
        .map_err(|error| format!("line {number}: {error}"))?;
    // Serde also supports positional struct representations. JSONL v1 requires
    // each record to be a named-field object, even when read directly from stdin.
    if !line.trim_start().starts_with('{') {
        return Err(format!("line {number}: expected a JSON object").into());
    }
    serde_json::from_str(&line).map_err(|error| format!("line {number}: {error}").into())
}

pub fn replay(
    input: impl BufRead,
    mut telemetry: impl Write,
    mut response: impl Write,
    policy: Policy,
) -> Result<()> {
    let mut lines = input.lines();
    let Input::Start(source) = parse(lines.next(), 1)? else {
        return Err("expected a start record on line 1".into());
    };
    source.validate()?;
    emit(
        &mut telemetry,
        Event::Start {
            source: &source,
            threshold: policy.threshold,
        },
    )?;

    let mut processed = 0;
    let status = loop {
        match parse(lines.next(), processed + 2)? {
            Input::Start(_) => return Err("unexpected second start record".into()),
            Input::End { token_count } => {
                if token_count != processed {
                    return Err(
                        format!("end count {token_count} differs from {processed} tokens").into()
                    );
                }
                if lines.next().is_some() {
                    return Err("unexpected input after end record".into());
                }
                break "complete";
            }
            Input::Token(token) => {
                if token.index != processed || processed >= MAX_SAFE_INTEGER {
                    return Err(format!(
                        "invalid token index {}, expected {processed}",
                        token.index
                    )
                    .into());
                }
                if token
                    .sae_feature_ids
                    .iter()
                    .flatten()
                    .any(|id| *id > MAX_SAFE_INTEGER)
                {
                    return Err("SAE feature ID exceeds the safe integer range".into());
                }
                let decision = policy.decide(token.risk)?;
                if decision == Decision::Allow {
                    response.write_all(token.text.as_bytes())?;
                    response.flush()?;
                }
                emit(
                    &mut telemetry,
                    Event::Decision {
                        trace_id: &source.trace_id,
                        token: &token,
                        decision,
                    },
                )?;
                processed += 1;
                if decision == Decision::Halt {
                    break "halted";
                }
            }
        }
    };
    emit(
        &mut telemetry,
        Event::End {
            trace_id: &source.trace_id,
            status,
            token_count: processed,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const START: &str = r#"{"type":"start","schema_version":1,"trace_id":"T","model_id":"M","model_revision":"V","probe_id":"P","probe_revision":"R","layer":0}"#;
    const ALLOW: &str = r#"{"type":"token","index":0,"text":" café\n","risk":0.125}"#;
    const HALT: &str = r#"{"type":"token","index":1,"text":"withheld","risk":0.5,"sae_feature_ids":[7]}"#;

    fn run(records: &[&str]) -> (Result<()>, String, Vec<Value>) {
        let input = records.join("\r\n");
        let mut response = Vec::new();
        let mut telemetry = Vec::new();
        let result = replay(
            input.as_bytes(),
            &mut telemetry,
            &mut response,
            Policy::new(0.5).unwrap(),
        );
        let events = String::from_utf8(telemetry)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        (result, String::from_utf8(response).unwrap(), events)
    }

    #[test]
    fn halt_reports_trigger_and_never_reads_or_releases_the_suffix() {
        let (result, response, events) = run(&[START, ALLOW, HALT, "unread invalid record"]);
        result.unwrap();
        assert_eq!(response, " café\n");
        assert_eq!(events.len(), 4);
        assert_eq!(events[0]["schema_version"], 1);
        assert_eq!(events[0]["threshold"], 0.5);
        assert_eq!(events[1]["decision"], "ALLOW");
        assert!(events[1].get("sae_feature_ids").is_none());
        assert_eq!(events[2], json!({
            "type": "decision", "trace_id": "T", "text": "withheld", "index": 1,
            "risk": 0.5, "decision": "HALT", "sae_feature_ids": [7]
        }));
        assert_eq!(events[3], json!({
            "type": "end", "trace_id": "T", "status": "halted", "token_count": 2
        }));
    }

    #[test]
    fn permits_empty_and_all_allow_traces() {
        let (result, response, events) = run(&[START, r#"{"type":"end","token_count":0}"#]);
        result.unwrap();
        assert!(response.is_empty());
        assert_eq!(events[1]["status"], "complete");
        let (result, response, events) = run(&[START, ALLOW, r#"{"type":"end","token_count":1}"#]);
        result.unwrap();
        assert_eq!(response, " café\n");
        assert_eq!(events[2]["token_count"], 1);
        assert_eq!(events[2]["status"], "complete");
    }

    #[test]
    fn bad_stream_cannot_emit_a_successful_end() {
        for records in [
            vec![],
            vec![ALLOW],
            vec![START, START],
            vec![START, ALLOW],
            vec![START, HALT],
            vec![START, ALLOW, r#"{"type":"end","token_count":0}"#],
            vec![START, r#"{"type":"end","token_count":0}"#, ALLOW],
        ] {
            let (result, _, events) = run(&records);
            assert!(result.is_err());
            assert!(events.iter().all(|event| event["type"] != "end"));
        }
    }

    #[test]
    fn rejects_malformed_or_ambiguous_records_before_release() {
        for token in [
            r#"["token",0,"x",0,[]]"#,
            r#"{"type":"token","index":0,"text":"x","risk":1.1}"#,
            r#"{"type":"token","index":false,"text":"x","risk":0}"#,
            r#"{"type":"token","index":0.0,"text":"x","risk":0}"#,
            r#"{"type":"token","index":0,"text":"x","risk":0,"risk":1}"#,
            r#"{"type":"token","index":0,"text":"x","risk":0,"unknown":1}"#,
            r#"{"type":"token","index":0,"text":"x","risk":0,"sae_feature_ids":null}"#,
            r#"{"type":"token","index":0,"text":"x","risk":0,"sae_feature_ids":[9007199254740992]}"#,
        ] {
            let (result, response, events) = run(&[START, token]);
            assert!(result.is_err());
            assert!(response.is_empty());
            assert_eq!(events.len(), 1, "{token}");
        }
        for bad_start in [
            START.replace(r#""schema_version":1"#, r#""schema_version":2"#),
            START.replace(r#""model_id":"M""#, r#""model_id":"  ""#),
            START.replace(r#""model_id":"M""#, r#""model_id":"\u001c""#),
            START.replace(r#""layer":0"#, r#""layer":9007199254740992"#),
        ] {
            let (result, response, events) = run(&[&bad_start]);
            assert!(result.is_err());
            assert!(response.is_empty());
            assert!(events.is_empty());
        }
    }
}
