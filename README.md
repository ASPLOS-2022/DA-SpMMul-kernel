# compile

change Makefile first line: `GPUARCH = sm_70` to `GPUARCH = sm_[YOUR_GPU_CC]`

```
make
```
# run
`bash run.sh input-file-name [more files]`
example `bash run.sh amazon0302.mtx amazon0312.mtx `

Output dumped to log.csv.