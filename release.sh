#!/bin/bash

# 安装 bump2version 和 twine
pip3 install --user bump2version twine

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
    ;;
  2)
    VERSION_TYPE="minor"
    ;;
  3)
    VERSION_TYPE="major"
    ;;
  *)
    echo "无效选择，退出脚本。"
    exit 1
    ;;
esac

# 更新版本号
python3 -m bumpversion $VERSION_TYPE --allow-dirty

# 获取新的版本号
NEW_VERSION=$(python3 setup.py --version)

# 提交到 Git 并打标签
git add .
git commit -m "Bump version to $NEW_VERSION"
git tag "v$NEW_VERSION"
git push origin --tags

# 创建分发档案
python3 setup.py sdist bdist_wheel

# 上传到 PyPI
python3 -m twine upload dist/*

echo "版本 $NEW_VERSION 已成功发布到 PyPI。"

