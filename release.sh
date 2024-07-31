#!/bin/bash

# 安装 versioneer 和 bump2version
pip3 install versioneer bump2version

# 初始化 versioneer
versioneer install

# 获取用户输入
echo "请选择版本更新类型:"
echo "1. patch"
echo "2. minor"
echo "3. major"
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
bump2version $VERSION_TYPE

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

