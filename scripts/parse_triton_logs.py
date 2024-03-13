import re
from collections import defaultdict
import csv
import re
import csv
from collections import defaultdict

def parse_triton_logs(log_content):
    metrics = []
    log_sections = re.split(r'Collected metrics at .*', log_content)

    for section in log_sections:
        if not section.strip():
            continue
        
        cur_metrics = {}

        # Parse KV cache metrics
        # Parse KV cache metrics
        kv_cache_metrics = re.findall(r'nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="(.*?)",.*?} (\d+)', section)
        for metric_type, value in kv_cache_metrics:
            metric_type = "kv_cache_" + metric_type
            cur_metrics[metric_type] = int(value)

        # # # Parse successful inference requests
        # #  # Parse successful inference requests
        # success_metrics = re.findall(r'nv_inference_request_success{.*?} (\d+)', section)
        # cur_metrics['successful_requests'] = sum(map(int, success_metrics)) if success_metrics else 0  # Change here
        # # metrics['successful_requests'] = sum(map(int, success_metrics)) if success_metrics else 0  # Change here


        # nv_inference_request_success{model="ensemble",version="1"} 0
# nv_inference_request_success{model="tensorrt_llm",version="1"} 0
# nv_inference_request_success{model="tensorrt_llm_bls",version="1"} 0
# nv_inference_request_success{model="preprocessing",version="1"} 0
# nv_inference_request_success{model="postprocessing",version="1"} 0
 
        # # # Parse successful inference requests
        for model in ['ensemble', 'tensorrt_llm', 'tensorrt_llm_bls', 'preprocessing', 'postprocessing']:
            success_metrics = re.findall(r'nv_inference_request_success{model="%s",version="1"} (\d+)' % model, section)
            cur_metrics['successful_requests_' + model] = sum(map(int, success_metrics)) if success_metrics else 0
       
        for model in ['tensorrt_llm', 'preprocessing', 'postprocessing']:
            success_metrics = re.findall(r'nv_inference_request_failure{model="%s",version="1"} (\d+)' % model, section)
            cur_metrics['failed_requests_' + model] = sum(map(int, success_metrics)) if success_metrics else 0
       
        for model in ['tensorrt_llm', 'preprocessing', 'postprocessing']:
            success_metrics = re.findall(r'nv_inference_pending_request_count{model="%s",version="1"} (\d+)' % model, section)
            cur_metrics['pending_requests_' + model] = sum(map(int, success_metrics)) if success_metrics else 0
        
        for model in ['generation_requests', 'total_context_tokens']:
            # inflight metrics
            success_metrics = re.findall(r'nv_trt_llm_inflight_batcher_metrics{inflight_batcher_specific_metric="%s",.*?} (\d+)' % model, section)
            cur_metrics[model] = sum(map(int, success_metrics)) if success_metrics else 0

        for stage in ["context", "scheduled", "max", "active"]:
            success_metrics = re.findall(r'nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="%s",version="1"} (\d+)' % stage, section)
            cur_metrics[stage + "_requests"] = sum(map(int, success_metrics)) if success_metrics else 0
    

        metrics.append(cur_metrics)
    
    return metrics

def write_metrics_to_csv(metrics, output_file):
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        log_content = f.read()
    metrics = parse_triton_logs(log_content)
    output_file = sys.argv[2]
    write_metrics_to_csv(metrics, output_file)