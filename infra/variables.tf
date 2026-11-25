variable "kubeconfig_path" {
  description = "Path to your kubeconfig file."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kube context to deploy into (null = current context)."
  type        = string
  default     = null
}

variable "namespace" {
  description = "Namespace for the platform."
  type        = string
  default     = "cricket-mlops"
}

variable "image" {
  description = "Container image for the serving API."
  type        = string
  default     = "cricket-mlops:latest"
}

variable "storage_size" {
  description = "Size of the volume holding runs, registry and audit log."
  type        = string
  default     = "1Gi"
}

variable "min_replicas" {
  description = "Minimum serving pods."
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum serving pods."
  type        = number
  default     = 5
}

variable "cpu_target_percent" {
  description = "Average CPU utilisation that triggers scale-out."
  type        = number
  default     = 70
}
