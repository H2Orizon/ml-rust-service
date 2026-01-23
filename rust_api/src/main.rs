use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use onnxruntime::{
    GraphOptimizationLevel, environment::Environment, ndarray::Array2, tensor::OrtOwnedTensor
};


#[derive(Deserialize)]
struct PredictRequest{
    input: Vec<i64>,
}

#[derive(Serialize)]
struct PredictResponse{
    prediction: f32,
}

#[derive(Clone)]
struct AppState {
    env: Arc<Environment>,
}

async fn predict(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Json<PredictResponse> {
    let env = &state.env;

    let mut session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file("../python_ml/data/models/model.onnx")
        .unwrap();

    let input_array = 
        Array2::from_shape_vec((1, 200), req.input.iter()
            .map(|&x| x as i32).collect::<Vec<i32>>())
            .unwrap();

    let outputs: Vec<OrtOwnedTensor<f32, _>> =
        session.run(vec![input_array.into()]).unwrap();

    let prediction = outputs[0][[0,0]];

    Json(PredictResponse {prediction})
}

#[tokio::main]
async fn main() {
    let environment  = Arc::new(
        Environment::builder()
            .with_name("ml")
            .build()
            .unwrap(),
    );

    let state = AppState { env: environment };

    let app = Router::new()
        .route("/predict", post(predict))
        .with_state(state);

    println!("Rust ML API running on http://localhost:8000");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}
