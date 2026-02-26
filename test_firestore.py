"""Quick test for Firestore connection."""
from google.cloud import firestore

PROJECT_ID = "queryquest-1771952465"

try:
    db = firestore.Client(project=PROJECT_ID)

    # Create test document
    doc_ref = db.collection("agent_sessions").document("_test")
    doc_ref.set({"test": True, "message": "Firestore is working!"})

    # Read it back
    doc = doc_ref.get()
    print(f"✅ Firestore connected!")
    print(f"   Data: {doc.to_dict()}")

    # Create HITL collection
    hitl_ref = db.collection("hitl_checkpoints").document("_test")
    hitl_ref.set({"test": True})
    print(f"✅ Created hitl_checkpoints collection")

    print("\n🎉 Firestore is ready!")

except Exception as e:
    print(f"❌ Firestore error: {e}")
