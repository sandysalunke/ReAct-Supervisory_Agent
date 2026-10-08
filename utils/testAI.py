from utils.tracing import tracer
import time

with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test.message", "Hello from Multi-Agent System")
    time.sleep(1)

print("Trace sent")