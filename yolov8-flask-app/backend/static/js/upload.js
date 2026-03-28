document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const imageFile = document.getElementById('image-upload').files[0];
    const modelSelect = document.getElementById('model-select');
    const customModelUpload = document.getElementById('custom-model-upload');
    
    if (!imageFile) {
        alert('请选择一张图片');
        return;
    }
    
    formData.append('image', imageFile);
    
    // 检查是否上传了自定义模型
    if (customModelUpload.files.length > 0) {
        formData.append('custom_model', customModelUpload.files[0]);
    } else {
        // 使用预定义模型
        formData.append('model_path', modelSelect.value);
    }
    
    // 显示上传的图像
    const uploadedImage = document.getElementById('uploaded-image');
    uploadedImage.innerHTML = `<img src="${URL.createObjectURL(imageFile)}" alt="上传的图像">`;
    
    try {
        // 发送请求到后端
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // 跳转到结果页面
        window.location.href = `/result/${result.id}`;
    } catch (error) {
        console.error('检测过程中发生错误:', error);
        alert('检测过程中发生错误，请查看控制台获取详细信息');
    }
});