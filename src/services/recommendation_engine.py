class RecommendationEngine:
    """
    Knowledge-base Recommendation and Alert Layer
    """
    def __init__(self):
        pass
        
    def evaluate(self, current_metrics, predictions, anomaly_flags):
        """
        current_metrics: dict containing CPU_usage, Memory_usage, Request_rate, Response_time, Error_rate
        predictions: dict containing future probability or future values
        anomaly_flags: dict containing is_high_conf_anomaly
        """
        cpu = current_metrics.get('CPU_usage', 0)
        mem = current_metrics.get('Memory_usage', 0)
        req = current_metrics.get('Request_rate', 0)
        lat = current_metrics.get('Response_time', 0)
        err = current_metrics.get('Error_rate', 0)
        
        congestion_prob = predictions.get('Congestion_probability', 0)
        is_anomaly = anomaly_flags.get('is_high_conf_anomaly', False)
        
        # 1. Alert Level Logic
        alert_level = "Normal"
        if congestion_prob > 0.8 or is_anomaly:
            alert_level = "Critical"
        elif congestion_prob > 0.6:
            alert_level = "Warning"
            
        # 2. Recommendation Knowledge Base Logic
        recommendations = []
        inference = "System operating normally."
        
        # Pattern 1: Traffic Congestion
        # Condition: CPU high, Requests increasing, Latency rising
        if cpu > 80 and req > 1000 and lat > 100:
            inference = "Traffic congestion detected due to high load."
            recommendations.extend([
                "Horizontal scaling: Increase replica count.",
                "Enable/Adjust Load Balancing.",
                "Enable request caching for static assets."
            ])
            
        # Pattern 2: Memory Leak / Resource exhaustion
        # Condition: Memory high, Error rate rising
        elif mem > 85 and err > 5:
            inference = "Memory leak possibility or insufficient memory allocation."
            recommendations.extend([
                "Restart service instances to flush memory.",
                "Increase memory allocation limit (Vertical scaling).",
                "Investigate application heap dumps."
            ])
            
        # Other simple rules
        elif is_anomaly and alert_level == "Critical":
            inference = "Unknown critical anomaly detected."
            recommendations.append("Investigate system logs immediately.")
            
        if not recommendations:
            recommendations.append("No immediate action required.")
            
        return {
            "Alert_Level": alert_level,
            "Inference": inference,
            "Recommendations": recommendations
        }

if __name__ == "__main__":
    # Test rules
    engine = RecommendationEngine()
    current = {'CPU_usage': 90, 'Memory_usage': 60, 'Request_rate': 1500, 'Response_time': 200, 'Error_rate': 0}
    preds = {'Congestion_probability': 0.85}
    flags = {'is_high_conf_anomaly': True}
    
    result = engine.evaluate(current, preds, flags)
    print(result)
