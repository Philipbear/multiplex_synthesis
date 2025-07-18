# Install required packages if not already installed
if (!require("circlize")) install.packages("circlize")
if (!require("RColorBrewer")) install.packages("RColorBrewer")
if (!require("viridis")) install.packages("viridis")

# Load libraries
library(circlize)
library(RColorBrewer)
library(viridis)

# Read the combined summary data
df <- read.csv("/Users/shipei/Documents/projects/multiplex_synthesis/library_analysis/cmpd_class/data/cmpd_class_summary.tsv", sep = "\t")

# Rename columns to standard names
colnames(df) <- c("class_1", "class_2", "count")

# Filter data with count
# df <- df[df$count > 50,]

# Get unique classes from both columns automatically
all_classes <- unique(c(df$class_1, df$class_2))
class_order <- sort(all_classes)  # Sort alphabetically for consistency

# Create a matrix for the chord diagram
# Initialize an empty matrix with zeros
chord_matrix <- matrix(0, nrow = length(class_order), ncol = length(class_order))
rownames(chord_matrix) <- class_order
colnames(chord_matrix) <- class_order

# Fill the matrix with counts from the data
for (i in 1:nrow(df)) {
  # Check if both classes are in our filtered list
  if (df$class_1[i] %in% class_order && df$class_2[i] %in% class_order) {
    row_idx <- which(class_order == df$class_1[i])
    col_idx <- which(class_order == df$class_2[i])
    
    # Add the count to the matrix
    chord_matrix[row_idx, col_idx] <- df$count[i]
    
    # Make the matrix symmetric (if the same class appears on both sides)
    if (row_idx != col_idx) {
      chord_matrix[col_idx, row_idx] <- df$count[i]
    }
  }
}

# Generate discrete colors using qualitative palettes
n_classes <- length(class_order)

# Option 1: Use RColorBrewer qualitative palettes
if (n_classes <= 12) {
  colors <- brewer.pal(min(n_classes, 12), "Set3")
} else {
  # For more than 12 classes, combine multiple palettes
  colors <- c(
    brewer.pal(12, "Set3"),
    brewer.pal(min(n_classes - 12, 8), "Set2"),
    brewer.pal(min(n_classes - 20, 9), "Set1")
  )[1:n_classes]
}

# Alternative discrete color options:
# colors <- brewer.pal(min(n_classes, 12), "Paired")
# colors <- brewer.pal(min(n_classes, 8), "Dark2")
# colors <- brewer.pal(min(n_classes, 9), "Set1")
# colors <- brewer.pal(min(n_classes, 8), "Set2")
# colors <- brewer.pal(min(n_classes, 12), "Set3")

# Option 2: Custom discrete color palette
# colors <- c("#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A", 
#            "#B15928", "#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F",
#            "#CAB2D6", "#FFFF99", "#05696B", "#8B4513", "#FF69B4")[1:n_classes]

# Create named vector for sector colors
sector_colors <- setNames(colors[1:n_classes], class_order)

# Function to create the chord diagram
create_chord_diagram <- function() {
  # Reset the circos parameters
  circos.clear()
  
  # Create the chord diagram with customized parameters
  chordDiagram(
    chord_matrix,
    grid.col = sector_colors,
    transparency = 0.25,
    directional = FALSE,
    annotationTrack = "grid",
    preAllocateTracks = list(track.height = 0.1),
    annotationTrackHeight = c(0.05, 0.1)
  )
  
  # Add legend positioned outside the plot area
  legend(x=0.95, y=0.75,
         legend = names(sector_colors), 
         fill = sector_colors, 
         border = NA,
         bty = "n",  # no box around legend
         cex = 0.7,  # smaller text size
         x.intersp = 0.3,  # horizontal spacing
         y.intersp = 1.15,  # vertical spacing
         xjust = 0,  # left-justify the legend
         ncol = 1)
}

# Create the SVG version with transparent background and smaller size
svg("chord_diagram.svg", width = 12, height = 3.5, bg = "transparent")
par(mfrow = c(1,1), mar = c(1, 1, 1, 1))
create_chord_diagram()
dev.off()

# Print confirmation message
cat("Chord diagram saved as chord_diagram_np_pathways.svg\n")
cat("Classes found:", paste(class_order, collapse = ", "), "\n")
cat("Number of classes:", length(class_order), "\n")