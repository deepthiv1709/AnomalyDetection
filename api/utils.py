def format_response(pred, score):
    return {
        "prediction": int(pred),
        "risk_score": float(score),
        "decision": "Review" if score < 0 else "Approve"
    }