import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent.multi_agent import MultiAgentOrchestrator

logging.basicConfig(level=logging.INFO)

query = "Find researchers actively publishing on new treatments for idiopathic pulmonary fibrosis in the last 3 years."
orchestrator = MultiAgentOrchestrator()
print(f"Session ID: {orchestrator.session_id}")

print("Sending first query...")
response1 = orchestrator.process_message(query)
print("\n--- RESPONSE 1 ---")
# Save output to file to avoid console encoding issues
with open("test_output.md", "w", encoding="utf-8") as f:
    f.write(response1['message'])
print("Output saved to test_output.md")

print("\nSending 'yes' to proceed (step 1)...")
response2 = orchestrator.process_message("yes")
print("\n--- RESPONSE 2 ---")
# Save output to file to avoid console encoding issues
with open("test_output.md", "a", encoding="utf-8") as f: # Use 'a' for append
    f.write("\n\n--- RESPONSE 2 ---\n")
    f.write(response2['message'])
print("Output appended to test_output.md")

print("\nSending 'yes' to proceed (step 2)...")
response3 = orchestrator.process_message("yes")
print("\n--- RESPONSE 3 ---")
# Save output to file to avoid console encoding issues
with open("test_output.md", "a", encoding="utf-8") as f: # Use 'a' for append
    f.write("\n\n--- RESPONSE 3 ---\n")
    f.write(response3['message'])
print("Output appended to test_output.md")

print("\nSending 'yes' to proceed (step 3)...")
response4 = orchestrator.process_message("yes")
print("\n--- RESPONSE 4 ---")
# Save output to file to avoid console encoding issues
with open("test_output.md", "a", encoding="utf-8") as f: # Use 'a' for append
    f.write("\n\n--- RESPONSE 4 ---\n")
    f.write(response4['message'])
print("Output appended to test_output.md")
