import random
import time


# ============================================================
# LOAD BALANCER CONFIGURATION
# ============================================================

CHAMPION_PERCENTAGE = 80
CHALLENGER_PERCENTAGE = 20

# ============================================================
# METRICS STORAGE
# ============================================================

metrics = {
    "champion_requests": 0,
    "challenger_requests": 0,
    "champion_latency": [],
    "challenger_latency": [],
}


# ============================================================
# ROUTING LOGIC
# ============================================================

def choose_model():
    """
    Returns either:
    champion
    challenger
    """

    random_number = random.randint(1, 100)

    if random_number <= CHAMPION_PERCENTAGE:
        metrics["champion_requests"] += 1
        return "champion"

    metrics["challenger_requests"] += 1
    return "challenger"


# ============================================================
# LATENCY LOGGER
# ============================================================

def log_latency(model_name: str, latency: float):

    if model_name == "champion":
        metrics["champion_latency"].append(latency)

    else:
        metrics["challenger_latency"].append(latency)


# ============================================================
# METRICS
# ============================================================

def get_metrics():

    champion_avg = (
        sum(metrics["champion_latency"])
        / len(metrics["champion_latency"])
        if metrics["champion_latency"]
        else 0
    )

    challenger_avg = (
        sum(metrics["challenger_latency"])
        / len(metrics["challenger_latency"])
        if metrics["challenger_latency"]
        else 0
    )

    total = (
        metrics["champion_requests"]
        + metrics["challenger_requests"]
    )

    return {
        "routing_strategy": "80/20 A/B Testing",
        "total_requests": total,
        "champion_requests": metrics["champion_requests"],
        "challenger_requests": metrics["challenger_requests"],
        "champion_avg_latency_ms": round(champion_avg * 1000, 2),
        "challenger_avg_latency_ms": round(challenger_avg * 1000, 2),
    }