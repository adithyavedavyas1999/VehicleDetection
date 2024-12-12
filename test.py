# Evaluate on test images
results = model.val(plots=True)
print(f"Metrics: {results}")

# Set confidence threshold
conf_threshold = 0.5
results = model("/content/drive/MyDrive/archive (1)/TestVideo/TrafficPolice.mp4", conf=conf_threshold)
