document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const imageFile = document.getElementById('image-upload').files[0];
    
    if (!imageFile) {
        alert('请选择一张图片');
        return;
    }
    
    formData.append('image', imageFile);
    
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