output "namespace" {
  value = kubernetes_namespace_v1.platform.metadata[0].name
}

output "service" {
  value = kubernetes_service_v1.serving.metadata[0].name
}

output "try_it" {
  value = "kubectl -n ${kubernetes_namespace_v1.platform.metadata[0].name} port-forward svc/${kubernetes_service_v1.serving.metadata[0].name} 8000:80"
}
