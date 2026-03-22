# -*- coding: utf-8 -*-
"""
Internationalization (i18n) strings for the gesture recognition system.
Supports multiple languages with easy string lookup.
"""

from typing import Dict, Optional

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'zh']

# String definitions for all supported languages
STRINGS: Dict[str, Dict[str, str]] = {
    'en': {
        # Application
        'app_title': 'Gesture Recognition System based on YOLOv8/v5',
        'app_version': 'Current version: v1.0',
        'version_title': 'Version Info',
        
        # Exit dialog
        'exit_title': 'Gesture Recognition System',
        'exit_message': 'Are you sure you want to exit?',
        
        # Camera
        'camera_started': 'Real-time camera started',
        'camera_error': 'Please check if the camera is connected correctly!',
        
        # File operations
        'select_model': 'Select Model File',
        'select_image': 'Select Image File',
        'select_video': 'Select Video File',
        'select_folder': 'Select Folder',
        'model_selected': ' selected',
        'path_selected': ' path selected',
        'file_selected': ' file selected',
        'use_default_model': 'Using default model',
        
        # Detection
        'all_targets': 'All Targets',
        'starting_recognition': 'Starting recognition system...\n\nLoading',
        'cannot_reproduce': 'Cannot reproduce current view, please select image record!',
        
        # Save operations
        'save_file': 'Save File',
        'save_start': 'Click OK to start saving...',
        'save_start_title': 'Start Saving',
        'save_image_success': '\nSuccess!\nImage file has been saved!',
        'save_video_success': '\nSuccess!\nVideo file has been saved!\nTable data saved as CSV!',
        'save_failed': 'Saving...\nFailed!\nPlease perform recognition before saving!',
        
        # Errors
        'invalid_path': 'Path is not a valid file or folder',
        'image_open_failed': 'Failed to open image file: {}',
        'video_open_failed': 'Failed to open video file: {}',
        
        # Labels
        'class': 'Class',
        'confidence': 'Confidence',
        'coordinates': 'Coordinates',
        'inference_time': 'Inference Time',
        'total_count': 'Total Count',
        
        # Login
        'username': 'Username',
        'password': 'Password',
        'verification_code': 'Verification Code',
        'login': 'Login',
        'register': 'Register',
        'change_password': 'Change Password',
        'change_avatar': 'Change Avatar',
        'logout': 'Logout Account',
        'logging_in': 'Logging in...',
        
        # Login errors
        'wrong_password': 'Incorrect password',
        'user_not_registered': 'User not registered',
        'user_already_exists': 'Username already exists',
        'password_too_short': 'Password too short',
        'select_avatar': 'Please select avatar file',
        'invalid_avatar': 'Invalid avatar file',
        'read_avatar_failed': 'Failed to read avatar',
        'file_not_exists': 'File does not exist',
        'valid_avatar': 'Valid avatar file',
        'incomplete_info': 'Incomplete information',
        'wrong_verification': 'Wrong verification code',
        'register_success': 'Registration successful',
        'password_changed': 'Password changed successfully',
        'avatar_changed': 'Avatar changed successfully',
        'account_deleted': 'Account deleted successfully',
        'username_not_exists': 'Username does not exist',
        
        # File types
        'image_files': 'Images (*.jpg;*.jpeg;*.png)',
        'video_files': 'Videos (*.mp4;*.avi)',
        'model_files': 'Model Files (*.pt)',
    },
    
    'zh': {
        # Application
        'app_title': '基于YOLOv8/v5的手势识别系统',
        'app_version': '当前版本为v1.0',
        'version_title': '版本信息',
        
        # Exit dialog
        'exit_title': '手势识别系统',
        'exit_message': '是否要退出程序？',
        
        # Camera
        'camera_started': '实时摄像已启动',
        'camera_error': '请检测摄像头与电脑是否连接正确！',
        
        # File operations
        'select_model': '选取模型文件',
        'select_image': '选取图片文件',
        'select_video': '选取视频文件',
        'select_folder': '选取文件夹',
        'model_selected': ' 已选中',
        'path_selected': ' 路径已选中',
        'file_selected': ' 文件已选中',
        'use_default_model': '使用默认模型',
        
        # Detection
        'all_targets': '所有目标',
        'starting_recognition': '正在启动识别系统...\n\nleading',
        'cannot_reproduce': '当前画面无法重现，请点选图片的识别记录！',
        
        # Save operations
        'save_file': '保存文件',
        'save_start': '请点击确定\n开始保存文件...',
        'save_start_title': '开始保存文件',
        'save_image_success': '\nSuccessed!\n当前图片文件已保存！',
        'save_video_success': '\nSuccessed!\n当前影像文件已保存！\n表格数据已保存为csv文件！',
        'save_failed': 'saving...\nFailed!\n请保存前先进行识别操作！',
        
        # Errors
        'invalid_path': '路径不是有效的文件或文件夹路径',
        'image_open_failed': '打开图像文件失败: {}',
        'video_open_failed': '打开视频文件失败: {}',
        
        # Labels
        'class': '类别',
        'confidence': '置信度',
        'coordinates': '坐标',
        'inference_time': '推理时间',
        'total_count': '目标总数',
        
        # Login
        'username': '用户名',
        'password': '密码',
        'verification_code': '验证码',
        'login': '登 录',
        'register': '注 册',
        'change_password': '修改密码',
        'change_avatar': '修改头像',
        'logout': '注销账户',
        'logging_in': '正在登录...',
        
        # Login errors
        'wrong_password': '密码不正确',
        'user_not_registered': '用户未注册',
        'user_already_exists': '该用户已被注册过',
        'password_too_short': '密码长度过短',
        'select_avatar': '请选择头像文件',
        'invalid_avatar': '无效头像文件',
        'read_avatar_failed': '读取头像失败',
        'file_not_exists': '文件不存在',
        'valid_avatar': '有效头像文件',
        'incomplete_info': '填写信息不全',
        'wrong_verification': '验证码错误',
        'register_success': '注册成功',
        'password_changed': '修改密码成功',
        'avatar_changed': '修改头像成功',
        'account_deleted': '账户已成功删除',
        'username_not_exists': '用户名不存在',
        
        # File types
        'image_files': '图片(*.jpg;*.jpeg;*.png)',
        'video_files': '视频(*.mp4;*.avi)',
        'model_files': 'Model File (*.pt)',
    }
}

# Current language setting
_current_language: str = 'en'


def set_language(lang: str) -> None:
    """
    Set the current language for string lookups.
    
    Args:
        lang: Language code ('en' or 'zh')
    """
    global _current_language
    if lang in SUPPORTED_LANGUAGES:
        _current_language = lang
    else:
        raise ValueError(f"Unsupported language: {lang}. Supported: {SUPPORTED_LANGUAGES}")


def get_language() -> str:
    """Get the current language setting."""
    return _current_language


def get_string(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """
    Get a localized string by key.
    
    Args:
        key: String key to look up
        lang: Optional language override (uses current language if not specified)
        **kwargs: Format arguments for string interpolation
        
    Returns:
        Localized string, or the key itself if not found
    """
    language = lang or _current_language
    
    if language not in STRINGS:
        language = 'en'  # Fallback to English
    
    string = STRINGS[language].get(key, key)
    
    # Apply format arguments if any
    if kwargs:
        try:
            string = string.format(**kwargs)
        except (KeyError, IndexError):
            pass
    
    return string


def _(key: str, **kwargs) -> str:
    """
    Shorthand alias for get_string().
    
    Args:
        key: String key to look up
        **kwargs: Format arguments
        
    Returns:
        Localized string
    """
    return get_string(key, **kwargs)
