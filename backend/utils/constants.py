# Master model configuration - single source of truth
MODELS = {
    # Free tier models

    "openai/claude-3-7-sonnet-20250219": {
        "aliases": ["claude-sonnet-37"],
        "pricing": {
            "input_cost_per_million_tokens": 3.00,
            "output_cost_per_million_tokens": 15.00
        },
        "tier_availability": ["free", "paid"]
    },
        
    "openai/deepseek/deepseek-chat": {
        "aliases": ["deepseek"],
        "pricing": {
            "input_cost_per_million_tokens": 0.38,
            "output_cost_per_million_tokens": 0.89
        },
        "tier_availability": ["free", "paid"]
    },
    "openai/qwen3-235b-a22b": {
        "aliases": ["qwen3"],
        "pricing": {
            "input_cost_per_million_tokens": 0.13,
            "output_cost_per_million_tokens": 0.60
        },
        "tier_availability": ["free", "paid"]
    },
    # Paid tier only models
    "openai/deepseek-chat-v3": {
        "aliases": ["deepseek-chat-v3"],
        "pricing": {
            "input_cost_per_million_tokens": 0.38,
            "output_cost_per_million_tokens": 0.89
        },
        "tier_availability": ["paid"]
    },
    "openai/gpt-4o": {
        "aliases": ["gpt-4o"],
        "pricing": {
            "input_cost_per_million_tokens": 2.50,
            "output_cost_per_million_tokens": 10.00
        },
        "tier_availability": ["paid"]
    }
}

# Derived structures (auto-generated from MODELS)
def _generate_model_structures():
    """Generate all model structures from the master MODELS dictionary."""
    
    # Generate tier lists
    free_models = []
    paid_models = []
    
    # Generate aliases
    aliases = {}
    
    # Generate pricing
    pricing = {}
    
    for model_name, config in MODELS.items():
        # Add to tier lists
        if "free" in config["tier_availability"]:
            free_models.append(model_name)
        if "paid" in config["tier_availability"]:
            paid_models.append(model_name)
        
        # Add aliases
        for alias in config["aliases"]:
            aliases[alias] = model_name
        
        # Add pricing
        pricing[model_name] = config["pricing"]
        
        # Also add pricing for legacy model name variations
        if model_name.startswith("openrouter/deepseek/"):
            legacy_name = model_name.replace("openrouter/", "")
            pricing[legacy_name] = config["pricing"]
        elif model_name.startswith("openrouter/qwen/"):
            legacy_name = model_name.replace("openrouter/", "")
            pricing[legacy_name] = config["pricing"]
        elif model_name.startswith("openrouter/google/"):
            legacy_name = model_name.replace("openrouter/", "")
            pricing[legacy_name] = config["pricing"]
        elif model_name.startswith("anthropic/"):
            # Add anthropic/claude-sonnet-4 alias for claude-sonnet-4-20250514
            if "claude-sonnet-4-20250514" in model_name:
                pricing["anthropic/claude-sonnet-4"] = config["pricing"]
    
    return free_models, paid_models, aliases, pricing

# Generate all structures
FREE_TIER_MODELS, PAID_TIER_MODELS, MODEL_NAME_ALIASES, HARDCODED_MODEL_PRICES = _generate_model_structures()

MODEL_ACCESS_TIERS = {
    "free": FREE_TIER_MODELS,
    "tier_2_20": PAID_TIER_MODELS,
    "tier_6_50": PAID_TIER_MODELS,
    "tier_12_100": PAID_TIER_MODELS,
    "tier_25_200": PAID_TIER_MODELS,
    "tier_50_400": PAID_TIER_MODELS,
    "tier_125_800": PAID_TIER_MODELS,
    "tier_200_1000": PAID_TIER_MODELS,
}
