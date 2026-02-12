# GitHub 仓库创建完整指南

## 一、准备工作

### 1. 确认项目位置

```
项目路径: /Users/Zhuanz/go/src/dl-from-scratch/
```

### 2. 确认 Git 历史

```bash
cd /Users/Zhuanz/go/src/dl-from-scratch
git log --oneline -5
```

应该看到：
```
* cd78330 chore: 重命名项目为 dl-from-scratch
* 69d0f18 feat: 添加 CNN/RNN/CUDA 示例和完整项目结构
* 626119e refactor: 重组项目结构为子项目
* ...
```

---

## 二、在 GitHub 创建仓库（手动）

### 步骤 1: 打开 GitHub

访问：https://github.com/new

### 步骤 2: 填写仓库信息

| 选项 | 值 |
|------|-----|
| **Repository name** | `dl-from-scratch` |
| **Description** | `深度学习从零实现：CNN、RNN、Transformer 与 CUDA 编程` |
| **Visibility** | ☐ Public / ☑ Private |
| **Initialize** | ⬜ 不勾选（已有代码）|
| **Add .gitignore** | ⬜ 不勾选 |
| **Choose license** | MIT License |

### 步骤 3: 点击 Create

创建后会显示：
```
https://github.com/forrestIsRunning/dl-from-scratch
```

---

## 三、本地关联远程仓库并推送

### 步骤 1: 添加远程仓库

```bash
cd /Users/Zhuanz/go/src/dl-from-scratch
git remote add origin https://github.com/forrestIsRunning/dl-from-scratch.git
```

### 步骤 2: 验证远程仓库

```bash
git remote -v
```

应该显示：
```
origin  https://github.com/forrestIsRunning/dl-from-scratch.git (fetch)
origin  https://github.com/forrestIsRunning/dl-from-scratch.git (push)
```

### 步骤 3: 推送代码

```bash
git push -u origin main
```

### 步骤 4: 验证推送成功

访问：https://github.com/forrestIsRunning/dl-from-scratch

---

## 四、可能遇到的问题

### 问题 1: 认证失败

**错误信息**：
```
remote: Support for password authentication was removed
```

**解决方法**：使用 Personal Access Token

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. 勾选：`repo` (全部权限）
4. 生成并复制 token
5. Push 时：
   - Username: forrestIsRunning
   - Password: `ghp_xxxxxxxxxxxxx` (你的 token)

### 问题 2: 远程仓库已存在

**错误信息**：
```
remote: Repository already exists
```

**解决方法**：强制推送
```bash
git push -u origin main --force
```

### 问题 3: 分支名称不是 main

**错误信息**：
```
error: src refspec main does not match any
```

**解决方法**：重命名分支
```bash
git branch -M main
git push -u origin main
```

---

## 五、推送后的配置

### 1. 设置仓库描述

编辑仓库的 `README.md` 或在 GitHub 页面上编辑：
```
Description: 深度学习从零实现：CNN、RNN、Transformer 与 CUDA 编程
Website: (可选)
Topics: pytorch, deep-learning, cnn, rnn, transformer, cuda, tutorial
License: MIT License
```

### 2. 设置仓库 Topics (标签)

在仓库页面 → Settings → Topics:
```
deep-learning
pytorch
cnn
rnn
lstm
transformer
cuda
machine-learning
tutorial
```

### 3. 添加 Star (自己 star)

方便以后快速访问你的项目！

---

## 六、完成后的 URL

```
GitHub: https://github.com/forrestIsRunning/dl-from-scratch
Clone: git clone https://github.com/forrestIsRunning/dl-from-scratch.git
```

---

## 快速命令汇总

```bash
# 完整流程
cd /Users/Zhuanz/go/src/dl-from-scratch
git remote add origin https://github.com/forrestIsRunning/dl-from-scratch.git
git push -u origin main

# 验证
git remote -v
git log --oneline -3
```
