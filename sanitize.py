import tensorflow as tf
import json

print("Loading original model...")
model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)

is_sequential = isinstance(model, tf.keras.Sequential)
config = model.get_config()

print("Sanitizing config...")
if is_sequential:
    for layer in config.get('layers', []):
        if layer['class_name'] == 'InputLayer':
            if 'batch_shape' in layer['config']:
                layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
            if 'optional' in layer['config']:
                layer['config'].pop('optional')
    clean_model = tf.keras.Sequential.from_config(config)
else:
    for layer in config.get('layers', []):
        if layer['class_name'] == 'InputLayer':
            if 'batch_shape' in layer['config']:
                layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
            if 'optional' in layer['config']:
                layer['config'].pop('optional')
    clean_model = tf.keras.Model.from_config(config)

print("Copying weights...")
clean_model.set_weights(model.get_weights())

print("Saving to .keras...")
clean_model.save('models/lstm_model.keras')
print("Done! Completely sanitized model generated.")
