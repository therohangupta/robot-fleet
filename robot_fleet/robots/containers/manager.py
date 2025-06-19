"""Container management for robot fleet"""
import os
import yaml
import docker
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime

@dataclass
class ContainerInfo:
    """Information about a running robot container
    
    Matches the ContainerInfo message in the proto definition:
    message ContainerInfo {
        string container_id = 1;
        string image = 2;
        string host = 3;
        int32 port = 4;
        map<string, string> environment = 5;
        google.protobuf.Timestamp created_at = 6;
    }
    """
    container_id: str
    image: str
    host: str
    port: int
    environment: Dict[str, str]
    created_at: Optional[str] = None  # ISO format timestamp string

class ContainerManager:
    """Manages Docker containers for robots"""
    
    def __init__(self):
        """Initialize the container manager"""
        self._clients: Dict[str, docker.DockerClient] = {}
        
    def _get_client(self, host: str, port: int) -> docker.DockerClient:
        """Get or create a Docker client for the specified host"""
        key = f"{host}:{port}"
        if key not in self._clients:
            if host == "localhost" or host == "unix":
                # For local Docker daemon, use Unix socket
                self._clients[key] = docker.from_env()
            else:
                # For TCP connections to remote Docker daemons
                base_url = f"tcp://{host}:{port}"
                self._clients[key] = docker.DockerClient(base_url=base_url)
        return self._clients[key]
        
    async def deploy_robot(self, robot_type: str, robot_id: str, deployment_info: dict, container_info: dict) -> ContainerInfo:
        """Deploy a robot container based on its configuration
        
        Args:
            robot_type: Type of robot to deploy
            robot_id: Unique identifier for the robot
            deployment_info: Docker deployment information (host, port)
            container_info: Container configuration (image, environment)
        """
        try:
            # Get deployment configuration
            host = deployment_info.get('docker_host', 'localhost')
            docker_port = deployment_info.get('docker_port', 2375)
            
            # Get container configuration
            image = container_info.get('image')
            container_env = container_info.get('environment', {})
            port = container_info.get('port', 5001)  # Get port from container_info
            
            # Get Docker client for the target host
            client = self._get_client(host, docker_port)
                
            # Prepare environment variables
            env = container_env.copy()
            env.update({
                'ROBOT_ID': robot_id,
                'PORT': str(port),
                'LOG_LEVEL': env.get('LOG_LEVEL', 'info')
            })
            
            # Run the container
            container = client.containers.run(
                name=robot_id,
                image=image,
                environment=env,
                ports={f'{port}/tcp': port},
                detach=True,
                command=container_info.get('command'),  # Use command if specified
                labels={
                    'robot_fleet': 'true',
                    'robot_type': robot_type
                }
            )
            
            return ContainerInfo(
                container_id=container.id,
                image=image,
                host=host,
                port=port,
                environment=env,
                created_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            print(f"Failed to start container: {str(e)}")
            raise
            
    async def stop_robot(self, robot_id: str, host: str, docker_port: int) -> bool:
        """Stop and remove a robot container"""
        try:
            client = self._get_client(host, docker_port)
            container = client.containers.get(robot_id)
            container.stop()
            container.remove()
            return True
        except docker.errors.NotFound:
            return False
        except Exception as e:
            print(f"Error stopping container: {str(e)}")
            return False
            
    async def list_robots(self) -> Dict[str, ContainerInfo]:
        """List all running robot containers across all hosts"""
        robots = {}
        
        # Ensure we have a localhost client
        if not self._clients:
            self._get_client("localhost", 0)
            
        for client_key, client in self._clients.items():
            host, port = client_key.split(':')
            try:
                containers = client.containers.list(
                    filters={'label': ['robot_fleet=true']}  # Docker expects a list of label filters
                )
                for container in containers:
                    env = container.attrs['Config']['Env']
                    robot_id = next(
                        (e.split('=')[1] for e in env if e.startswith('ROBOT_ID=')),
                        None
                    )
                    port = next(
                        (int(e.split('=')[1]) for e in env if e.startswith('PORT=')),
                        None
                    )
                    if robot_id and port:
                        # Convert environment list to dict
                        env_dict = {}
                        for env_str in env:
                            if '=' in env_str:
                                key, value = env_str.split('=', 1)
                                env_dict[key] = value
                                
                        robots[robot_id] = ContainerInfo(
                            container_id=container.id,
                            image=container.image.tags[0] if container.image.tags else "",
                            host=host,
                            port=port,
                            environment=env_dict,
                            created_at=container.attrs['Created']
                        )
            except Exception as e:
                print(f"Error listing containers on {host}:{port}: {str(e)}")
            
        return robots
        
    async def get_robot_logs(self, robot_id: str, host: str, docker_port: int) -> str:
        """Get logs from a robot container"""
        try:
            client = self._get_client(host, docker_port)
            container = client.containers.get(robot_id)
            return container.logs().decode('utf-8')
        except docker.errors.NotFound:
            return f"Container {robot_id} not found"
        except Exception as e:
            return f"Error getting logs: {str(e)}"
            
    async def check_image_exists(self, host: str, docker_port: int, image: str) -> bool:
        """Check if a Docker image exists locally or can be pulled from registry
        
        Args:
            host: Docker host
            docker_port: Docker port
            image: Image name (with optional tag)
            
        Returns:
            True if image exists locally or can be pulled, False otherwise
        """
        try:
            client = self._get_client(host, docker_port)
            
            # First check if image exists locally
            try:
                client.images.get(image)
                return True
            except docker.errors.ImageNotFound:
                # Image not found locally, try to check registry without pulling
                print(f"Image {image} not found locally, checking registry...")
                
                # Parse image name and tag
                if ':' in image:
                    repo, tag = image.split(':', 1)
                else:
                    repo, tag = image, 'latest'
                
                # Check if image exists in registry without pulling
                try:
                    registry_data = client.images.get_registry_data(image)
                    print(f"Image {image} found in registry with digest: {registry_data.id}")
                    return True
                except docker.errors.APIError as e:
                    # Some registries might require authentication or have rate limits
                    if "unauthorized" in str(e).lower():
                        print(f"Registry authentication required for {image}")
                        return False
                    if "not found" in str(e).lower():
                        print(f"Image {image} not found in registry")
                        return False
                    print(f"Registry error: {str(e)}")
                    return False
                except Exception as e:
                    print(f"Error checking registry: {str(e)}")
                    return False
        except Exception as e:
            print(f"Error checking image: {str(e)}")
            return False 