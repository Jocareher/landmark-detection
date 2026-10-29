# Ejecutar entrenamiento y TTA en Correfoc

La [guía de Correfoc de UPF](https://scc-doc.s.upf.edu/md_correfoc_cluster.html#slurm-queue-system) usa las particiones `std-gpu` (una NVIDIA L40S por nodo) y `high-gpu` (cuatro L40S por nodo). Este proyecto usa **una GPU por job**; `std-gpu` es la selección inicial. El launcher solicita `--gres=gpu:1`, conserva `sbatch` para enviar el job y usa `srun` dentro de la asignación. Comprueba `MaxTime` de la partición en vivo mediante `scontrol`, porque la guía no fija aquí un límite universal de tiempo.

## 1. Preparar datos, rutas y Python en el clúster

1. Entrá a Correfoc con tu cuenta UPF y ubicá el repositorio, datasets, pesos y prior PCA en rutas **absolutas compartidas**. La guía recomienda `/projects/lab_<laboratorio>/...` para datos y resultados importantes con copia de seguridad, y `/scratch/lab_<laboratorio>/...` para trabajo temporal intensivo. No asumimos que las rutas `/home/jreyes/...` de SNOW existan en Correfoc.
2. Prepará el entorno Python **en un nodo de cómputo**, no en el login. El launcher Correfoc usa un intérprete ya preparado (`python_executable`) y comprueba imports, versión Python y PyTorch CUDA antes de entrenar. No instala dependencias durante cada job. Requiere Python ≥ 3.11, `torch`, `torchvision` y los paquetes de `environments/requirements-hpc.txt`. Podés usar un entorno que ofrezca SCC o crear uno en tu espacio de proyecto. Si necesitás módulos, encontralos con `module spider Python`/`module spider PyTorch` y anotá los nombres exactos, en orden, en `modules`; Correfoc ofrece los stacks `modulepath/EESSI/2025.06` y `modulepath/UPF/apps`. La guía muestra como ejemplo `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`, pero no asumimos que ese módulo incluya todas nuestras dependencias.
3. Copiá `configs/correfoc_hpc.example.json` a `configs/correfoc_hpc.local.json`. Reemplazá todos los segmentos `REPLACE`, especialmente `runs_root`, `python_executable` y cada ruta de `paths`. El archivo local queda ignorado por Git. Si SCC te exige una cuenta Slurm, llená `account`; si no, dejalo en `null`.

`runs_root` debería estar en `projects` si querés conservar checkpoints y resultados. Usá `scratch` solo cuando tengas un procedimiento para copiar los resultados importantes a `projects`. La instalación de paquetes en los nodos puede depender del acceso a índices de Python; verificá eso con SCC si necesitás crear un entorno nuevo.

## 2. Validar y enviar

Ejecutá los wrappers desde el login. Preparan el snapshot del código, las rutas de logs y la llamada a `sbatch`; no ejecutes allí entrenamiento ni TTA.

```bash
cp configs/correfoc_hpc.example.json configs/correfoc_hpc.local.json
# Editar el JSON local con las rutas reales del clúster.
sinfo -o '%P %l %G'
scontrol show partition std-gpu

bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/correfoc_hpc.local.json \
  --config configs/normalizer_background_randomization.yaml \
  --dry-run

bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/correfoc_hpc.local.json \
  --config configs/normalizer_background_randomization.yaml \
  --epochs 1 --run-name correfoc_bg_smoke
```

La prueba de una época confirma el entorno y permite medir memoria y tiempo antes de lanzar las 60 épocas. Para otro entrenamiento, omití `--config` o elegí otro YAML. Para TTA:

```bash
bash slurm/tta_landmarks.sh \
  --settings configs/correfoc_hpc.local.json \
  --checkpoint /projects/lab_REPLACE/REPLACE_USER/landmark-detection/runs/REPLACE/checkpoints/full_model_best.pth \
  --dataset babyland --steps 1 --dry-run
```

Reemplazá la ruta del checkpoint antes de ejecutar. Quitá `--dry-run` tras revisar la orden `sbatch` impresa. `--partition high-gpu` es posible si `std-gpu` no satisface un requisito real; el código seguirá pidiendo **una sola GPU**. No uses `--gpu-type` en Correfoc: la documentación solicita el GRES genérico `gpu:1`.

Los logs quedan en `runs_root/<run>/logs/slurm-<jobid>.out` y `.err`. Para comprobar el estado: `squeue -u "$USER"`; al terminar: `sacct -j <jobid>` o `seff <jobid>`.

## Qué se cambió respecto a SNOW

| Aspecto | SNOW | Correfoc |
|---|---|---|
| Particiones GPU | `short`, `medium`, `high` según tiempo | `std-gpu` o `high-gpu` según hardware |
| GPU solicitada | `--gres=gpu:<tipo>:1` | `--gres=gpu:1` |
| Python | módulo Miniconda y entorno `lmks` creado/reparado por job | ruta de Python preparada y validada antes del trabajo |
| Datos y resultados | rutas de SNOW en `/home/...` | rutas reales de Correfoc en `projects`/`scratch`/`homes` |

Los scripts `scripts.main` y `scripts.evaluate`, así como las pérdidas PCA/Mahalanobis, no cambian. El cambio está en recursos, rutas y entorno de ejecución. Las rutas de los YAML se sustituyen por las de `paths` cuando se crea el job. El historial y metadatos de cada ejecución conservan el comando, partición, entorno y snapshot del código.
