// core/static/core/js/formToggle.js
function toggleInput(type) {
    const textInput = document.getElementById('text-input');
    const urlInput = document.getElementById('url-input');
    const inputTypeField = document.getElementById('input-type');

    if (type === 'text') {
        textInput.style.display = 'block';
        urlInput.style.display = 'none';
        inputTypeField.value = 'text';

        // Clear URL input value
        document.querySelector('input[name="url_content"]').value = '';
    } else if (type === 'url') {
        textInput.style.display = 'none';
        urlInput.style.display = 'block';
        inputTypeField.value = 'url';

        // Clear text input value
        document.querySelector('textarea[name="text_content"]').value = '';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form state
    const initialType = document.getElementById('id_input_type').value || 'text';
    toggleInput(initialType);

    // Add event listeners to buttons
    document.querySelectorAll('.toggle-btn').forEach(button => {
        button.addEventListener('click', function() {
            const type = this.dataset.inputType;
            toggleInput(type);
            
            // Clear previous errors
            const form = document.querySelector('form');
            form.querySelectorAll('.is-invalid').forEach(el => 
                el.classList.remove('is-invalid')
            );
        });
    });
});