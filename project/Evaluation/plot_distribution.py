import matplotlib.pyplot as plt

# Data for the Fitzpatrick skin type distribution
skin_types = [
    'Type I', 'Type II', 
    'Type III', 'Type IV', 'Type V', 'Type VI'
]
counts = [122, 223, 68, 44, 30, 25]

# Initialize the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(skin_types, counts, color='#4C72B0', edgecolor='black')

# Add the exact count above each bar for readability
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), 
             ha='center', va='bottom', fontweight='bold')

# Formatting and labels
plt.title('ISIC Dataset: Fitzpatrick Skin Type Distribution (Total: 512 Images)', fontsize=14)
plt.xlabel('Fitzpatrick Skin Type', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot to disk
plt.savefig('ISIC_Fitzpatrick_Distribution.png', dpi=300)