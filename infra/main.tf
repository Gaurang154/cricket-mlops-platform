# Provisions the platform on any Kubernetes cluster:
# kind / minikube on a laptop, or EKS / GKE / AKS in the cloud (just change the kube context).

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

locals {
  app = "cricket-serving"
}

resource "kubernetes_namespace_v1" "platform" {
  metadata {
    name = var.namespace
  }
}

# Persistent storage for runs, the model registry and the audit log
# (stands in for object storage + a metadata DB in a bigger deployment).
resource "kubernetes_persistent_volume_claim_v1" "state" {
  metadata {
    name      = "mlops-state"
    namespace = kubernetes_namespace_v1.platform.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = var.storage_size
      }
    }
  }
  wait_until_bound = false
}

# The serving API. On first boot it trains, gates and approves a bootstrap model.
resource "kubernetes_deployment_v1" "serving" {
  metadata {
    name      = local.app
    namespace = kubernetes_namespace_v1.platform.metadata[0].name
    labels = {
      app = local.app
    }
  }
  spec {
    replicas = var.min_replicas
    selector {
      match_labels = {
        app = local.app
      }
    }
    template {
      metadata {
        labels = {
          app = local.app
        }
      }
      spec {
        container {
          name              = "api"
          image             = var.image
          image_pull_policy = "IfNotPresent"
          port {
            container_port = 8000
          }
          env {
            name  = "MLOPS_HOME"
            value = "/data"
          }
          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }
          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
          volume_mount {
            name       = "state"
            mount_path = "/data"
          }
        }
        volume {
          name = "state"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.state.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "serving" {
  metadata {
    name      = local.app
    namespace = kubernetes_namespace_v1.platform.metadata[0].name
  }
  spec {
    selector = {
      app = local.app
    }
    port {
      port        = 80
      target_port = 8000
    }
  }
}

# Autoscaling: add pods when average CPU passes the target.
resource "kubernetes_horizontal_pod_autoscaler_v2" "serving" {
  metadata {
    name      = local.app
    namespace = kubernetes_namespace_v1.platform.metadata[0].name
  }
  spec {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment_v1.serving.metadata[0].name
    }
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.cpu_target_percent
        }
      }
    }
  }
}
