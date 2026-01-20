use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use onnxruntime::{
    GraphOptimizationLevel, environment::Environment, ndarray::Array2, tensor::OrtOwnedTensor
};


#[tokio::test]
async fn test_predict() {
    let client = reqwest::Client::new();
    let res = client.post("http://localhost:8000/predict")
        .json(&json!({"input": vec![0; 200]}))
        .send()
        .await
        .unwrap();

    assert!(res.status().is_success());
}