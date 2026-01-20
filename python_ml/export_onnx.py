import tensorflow as tf
import tf2onnx

models_path="python_ml/data/models/"

model = tf.keras.models.load_model(models_path+"model.h5")
print("Keras модель завантажена")

@tf.function(input_signature=[tf.TensorSpec([None, 200], tf.int32, name="input")])
def model_fn(x):
    return model(x)

output_path = models_path+"model.onnx"
model_proto, _ = tf2onnx.convert.from_function(
    model_fn,
    input_signature=[tf.TensorSpec([None, 200], tf.int32, name="input")],
    output_path=output_path,
    opset=13
)
print(f"✅ ONNX модель збережена в {output_path}")
