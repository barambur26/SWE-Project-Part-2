"""
src/api/routes/rate.py

POST /rate endpoint - Score a single model URL using Phase 1 metrics
"""
from flask import Blueprint, request, jsonify
import time
import logging
from typing import Dict, Any

# Import Phase 1 scoring components
from swe_project.core.exec_pool import run_parallel
from swe_project.core.scoring import combine
from swe_project.core.url_ctx import set_context, clear as clear_url_ctx
from swe_project.core.model_url import to_repo_id
from swe_project.metrics.base import registered
import re

bp = Blueprint('rate', __name__)

# Import metrics for registration side-effects
def _ensure_metrics_loaded():
    """Import all metric modules to trigger registration."""
    try:
        from swe_project.metrics import bus_factor  # noqa: F401
        from swe_project.metrics import code_quality  # noqa: F401
        from swe_project.metrics import dataset_and_code  # noqa: F401
        from swe_project.metrics import dataset_quality  # noqa: F401
        from swe_project.metrics import license  # noqa: F401
        from swe_project.metrics import performance_claims  # noqa: F401
        from swe_project.metrics import ramp_up_time  # noqa: F401
        from swe_project.metrics import size_score  # noqa: F401
    except ImportError as e:
        logging.error(f"Failed to import metrics: {e}")
        raise


@bp.route('/rate', methods=['POST'])
def rate_model():
    """
    Rate a single model from HuggingFace or GitHub.
    
    Request body:
    {
        "url": "https://huggingface.co/google-bert/bert-base-uncased",
        "code_url": "https://github.com/...",  # optional
        "dataset_url": "https://huggingface.co/datasets/..."  # optional
    }
    
    Response:
    {
        "name": "bert-base-uncased",
        "category": "MODEL",
        "net_score": 0.845,
        "net_score_latency": 1234,
        "ramp_up_time": 0.75,
        "ramp_up_time_latency": 456,
        ... (all 8 metrics with latencies)
    }
    """
    _ensure_metrics_loaded()
    
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing required field: url'}), 400
    
    model_url = data['url']
    code_url = data.get('code_url')
    dataset_url = data.get('dataset_url')
    
    # Set context for this model (Phase 1 pattern)
    clear_url_ctx()
    set_context(model_url, code_url, dataset_url)
    
    try:
        # Build tasks from registry
        tasks = []
        for _, field, compute in registered():
            def _task(func=compute, url=model_url):
                def run():
                    return func(url)
                return run
            tasks.append((field, _task()))
        
        # Run metrics in parallel (Phase 1 pattern)
        t0 = time.perf_counter()
        results = run_parallel(tasks, timeout_s=90)
        net_latency_ms = int((time.perf_counter() - t0) * 1000)
        
        # Extract values safely
        def _val(name: str) -> float:
            return float(results.get(name, {}).get('value', 0.0))
        
        def _lat(name: str) -> int:
            return int(results.get(name, {}).get('latency_ms', 0))
        
        # Handle size_score dict structure
        size_map = results.get('size_score', {}).get('value', {}) or {}
        for k in ('raspberry_pi', 'jetson_nano', 'desktop_pc', 'aws_server'):
            size_map.setdefault(k, 0.0)
        size_lat = _lat('size_score')
        
        # Gather scalars for net score calculation
        scalars = {
            'ramp_up_time': _val('ramp_up_time'),
            'bus_factor': _val('bus_factor'),
            'license': _val('license'),
            'dataset_and_code_score': _val('dataset_and_code_score'),
            'dataset_quality': _val('dataset_quality'),
            'code_quality': _val('code_quality'),
            'performance_claims': _val('performance_claims'),
            'size_score': sum(size_map.values()) / 4.0,
        }
        net_score = float(combine(scalars))
        
        # Extract model name from URL
        repo_id, _ = to_repo_id(model_url)
        model_name = re.split(r'[\\/]', repo_id.rstrip('/\\'))[-1]
        
        # Build response matching Phase 1 NDJSON structure
        response = {
            'name': model_name,
            'category': 'MODEL',
            'net_score': round(net_score, 3),
            'net_score_latency': net_latency_ms,
            'ramp_up_time': scalars['ramp_up_time'],
            'ramp_up_time_latency': _lat('ramp_up_time'),
            'bus_factor': scalars['bus_factor'],
            'bus_factor_latency': _lat('bus_factor'),
            'performance_claims': scalars['performance_claims'],
            'performance_claims_latency': _lat('performance_claims'),
            'license': scalars['license'],
            'license_latency': _lat('license'),
            'size_score': size_map,
            'size_score_latency': size_lat,
            'dataset_and_code_score': scalars['dataset_and_code_score'],
            'dataset_and_code_score_latency': _lat('dataset_and_code_score'),
            'dataset_quality': scalars['dataset_quality'],
            'dataset_quality_latency': _lat('dataset_quality'),
            'code_quality': scalars['code_quality'],
            'code_quality_latency': _lat('code_quality'),
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error rating model {model_url}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
