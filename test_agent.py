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
print(response1['message'])

print("\nSending 'yes' to proceed (step 1)...")
response2 = orchestrator.process_message("yes")
print("\n--- RESPONSE 2 ---")
print(response2['message'])

print("\nSending 'yes' to proceed (step 2)...")
response3 = orchestrator.process_message("yes")
print("\n--- RESPONSE 3 ---")
print(response3['message'])

print("\nSending 'yes' to proceed (step 3)...")
response4 = orchestrator.process_message("yes")
print("\n--- RESPONSE 4 ---")
print(response4['message'])
