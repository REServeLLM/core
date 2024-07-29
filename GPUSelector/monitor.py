from tritonclient.utils import InferenceServerException
from tritonclient.grpc import InferenceServerClient

def get_model_ranks(url="localhost:8081"):
    try:
        # Create a gRPC client for communicating with the server
        triton_client = InferenceServerClient(url=url)
        # Get the list of all models
        model_repository = triton_client.get_model_repository_index()
        model_ranks = {}

        for model in model_repository.models:
            model_name = model.name
            # Get the model metadata
            model_metadata = triton_client.get_model_metadata(model_name=model_name)
            print(f"Metadata for model {model_name}: {model_metadata}")
            model_ranks[model_name] = []

            # Try to get instance group from model config if not in metadata
            try:
                model_config = triton_client.get_model_config(model_name=model_name)
                print(f"Model config: {model_config}")
                instance_groups = model_config.instance_group
            except AttributeError:
                print(f"No instance_group found for model {model_name} in metadata or config")
                continue

            for instance_group in instance_groups:
                for instance in instance_group.instances:
                    # Get GPU ID
                    gpu_id = instance.gpus[0]
                    # Get instance ID
                    instance_id = instance.id
                    model_ranks[model_name].append((instance_id, gpu_id))

        return model_ranks

    except InferenceServerException as e:
        print(f"Error querying Triton: {e}")
        return None

model_ranks = get_model_ranks()
if model_ranks:
    for model_name, ranks in model_ranks.items():
        for instance_id, gpu_id in ranks:
            print(f"Model: {model_name}, Instance ID: {instance_id}, GPU ID: {gpu_id}")