#!/bin/bash
set -e

# 安装 bump2version 和 twine
echo "安装必要的工具..."
pip3 install --user bump2version twine >/dev/null 2>&1

# 获取当前版本号
CURRENT_VERSION=$(python3 setup.py --version)

# 计算目标版本号
PATCH_TARGET_VERSION=$(python3 -m bumpversion --dry-run --list patch --allow-dirty | grep new_version | sed -r s,"^.*=",,)
MINOR_TARGET_VERSION=$(python3 -m bumpversion --dry-run --list minor --allow-dirty | grep new_version | sed -r s,"^.*=",,)
MAJOR_TARGET_VERSION=$(python3 -m bumpversion --dry-run --list major --allow-dirty | grep new_version | sed -r s,"^.*=",,)

# 获取用户输入
echo "请选择版本更新类型:"
echo "1. patch  当前版本: $CURRENT_VERSION ---> 目标版本: $PATCH_TARGET_VERSION"
echo "2. minor  当前版本: $CURRENT_VERSION ---> 目标版本: $MINOR_TARGET_VERSION"
echo "3. major  当前版本: $CURRENT_VERSION ---> 目标版本: $MAJOR_TARGET_VERSION"
read -p "输入数字选择 (1, 2, 3): " choice

case $choice in
  1)
    VERSION_TYPE="patch"
    TARGET_VERSION=$PATCH_TARGET_VERSION
    ;;
  2)
    VERSION_TYPE="minor"
    TARGET_VERSION=$MINOR_TARGET_VERSION
    ;;
  3)
    VERSION_TYPE="major"
    TARGET_VERSION=$MAJOR_TARGET_VERSION
    ;;
  *)
    echo "无效选择，退出脚本。"
    exit 1
    ;;
esac

# 检查标签是否已存在
if git rev-parse "v$TARGET_VERSION" >/dev/null 2>&1; then
    read -p "警告: 标签 v$TARGET_VERSION 已存在。是否覆盖？(y/n): " overwrite
    if [[ $overwrite != "y" ]]; then
        echo "操作已取消。"
        exit 1
    fi
fi

# 更新版本号
python3 -m bumpversion $VERSION_TYPE --allow-dirty

# 获取新的版本号
NEW_VERSION=$(python3 setup.py --version)

# 确认 Git 操作
read -p "是否提交更改并推送到 Git？(y/n): " git_confirm
if [[ $git_confirm == "y" ]]; then
    git add .
    git commit -m "Bump version to $NEW_VERSION"
    git tag -f "v$NEW_VERSION"
    git push origin main --tags --force
    echo "更改已提交并推送到 Git。"
else
    echo "Git 操作已跳过。"
fi

# 创建分发档案
python3 setup.py sdist bdist_wheel

# 确认 PyPI 上传
read -p "是否上传到 PyPI？(y/n): " pypi_confirm
if [[ $pypi_confirm == "y" ]]; then
    # 检查是否已经存在相同版本
    if twine check dist/*$NEW_VERSION* >/dev/null 2>&1; then
        read -p "警告: PyPI 上可能已存在版本 $NEW_VERSION。是否继续上传？(y/n): " continue_upload
        if [[ $continue_upload != "y" ]]; then
            echo "PyPI 上传已取消。"
            exit 1
        fi
    fi
    python3 -m twine upload dist/*
    echo "版本 $NEW_VERSION 已成功发布到 PyPI。"
else
    echo "PyPI 上传已跳过。"
fi

# 清理临时文件
read -p "是否清理生成的分发文件？(y/n): " clean_confirm
if [[ $clean_confirm == "y" ]]; then
    rm -rf dist build *.egg-info
    echo "临时文件已清理。"
fi

echo "脚本执行完毕。"
