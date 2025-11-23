// Library page initialization
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scroll behavior
    document.querySelectorAll('.gesture-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });

});

