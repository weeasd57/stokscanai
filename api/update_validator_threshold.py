import pickle
import os

def update_threshold(model_path, new_threshold):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    
    old_threshold = artifact.get('approval_threshold', 'N/A')
    artifact['approval_threshold'] = new_threshold
    
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    
    print(f"✅ Updated {os.path.basename(model_path)} threshold: {old_threshold} -> {new_threshold}")

if __name__ == "__main__":
    model_path = r"c:\Users\MR_CODER\Desktop\AI stocks\api\models\COUNCIL_CRYPTO.pkl"
    update_threshold(model_path, 0.4)
