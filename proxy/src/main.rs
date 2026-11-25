mod policy;
mod wire;

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use policy::Policy;

const USAGE: &str = "firewall-proxy --threshold NUMBER --response NEW_FILE";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Options {
    policy: Policy,
    response: PathBuf,
}

fn options(mut args: impl Iterator<Item = OsString>) -> Result<Options> {
    let mut threshold = None;
    let mut response = None;
    while let Some(key) = args.next() {
        match key.to_str() {
            Some("--threshold") if threshold.is_none() => {
                let raw = args.next().ok_or("missing threshold value")?;
                let value = raw.to_str().ok_or("invalid threshold encoding")?.parse()?;
                threshold = Some(Policy::new(value)?);
            }
            Some("--response") if response.is_none() => {
                response = Some(PathBuf::from(args.next().ok_or("missing response path")?));
            }
            _ => return Err(format!("unknown or repeated option. Usage: {USAGE}").into()),
        }
    }
    Ok(Options {
        policy: threshold.ok_or("missing --threshold")?,
        response: response.ok_or("missing --response")?,
    })
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    let options = options(args.into_iter())?;
    let mut response = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(options.response)?;
    wire::replay(
        io::stdin().lock(),
        io::stdout().lock(),
        &mut response,
        options.policy,
    )
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("firewall-proxy: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options> {
        options(args.iter().map(OsString::from))
    }

    #[test]
    fn requires_both_a_valid_threshold_and_one_new_file_path() {
        assert!(parse(&["--response", "out.txt"]).is_err());
        assert!(parse(&["--threshold", "0.4"]).is_err());
        assert!(parse(&["--threshold", "NaN", "--response", "out.txt"]).is_err());
        assert!(parse(&["--threshold", "0.4", "--response", "a", "--response", "b"]).is_err());
        let options = parse(&["--response", "out.txt", "--threshold", "0.4"]).unwrap();
        assert_eq!(options.response, PathBuf::from("out.txt"));
        assert_eq!(options.policy.threshold, 0.4);
    }
}
