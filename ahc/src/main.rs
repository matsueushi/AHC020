use solver::*;

use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Request {
    input: String,
}

#[derive(Serialize)]
struct Response {
    req_id: String,
    output: String,
    score: usize,
    msg: String,
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    // let mut buf_reader = BufReader::new(event.payload.input.as_bytes());
    let mut source = proconio::source::once::OnceSource::from(event.payload.input.as_str());
    let input = Input::from_source(&mut source);
    let solution = solve(&input);

    let resp = Response {
        req_id: event.context.request_id,
        output: solution.output,
        score: solution.score,
        msg: format!(""),
    };

    Ok(resp)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        // disabling time is handy because CloudWatch will add the ingestion time.
        .without_time()
        .init();

    run(service_fn(function_handler)).await
}
