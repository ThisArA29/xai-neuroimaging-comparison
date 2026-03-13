library(tidyverse)
library(readxl)
library(glue)
library(scales)

# Root folder containing Excel files
root_dir <- "path/to/your/input/folder"

# Folder where plots will be saved
img_dir  <- file.path(root_dir, "Images")
dir.create(img_dir, showWarnings = FALSE, recursive = TRUE)

# Convert a string like "[1,2,3]" to a numeric vector
parse_bracket_list <- function(x) {
  if (is.null(x)) return(numeric(0))
  if (is.list(x)) x <- x[[1]]
  x <- as.character(x)
  x <- str_remove_all(x, "\\[|\\]")
  as.numeric(str_split(x, ",\\s*")[[1]])
}

# Convert a cell value to numeric vector
cell_to_numvec <- function(v) {
  if (is.null(v)) return(numeric(0))
  if (is.list(v)) v <- v[[1]]
  if (is.numeric(v)) return(as.numeric(v))
  v <- as.character(v)
  v <- str_remove_all(v, "\\[|\\]")
  as.numeric(str_split(v, ",\\s*")[[1]])
}

# 1. Biological validity polar heatmap
plot_bio_validity_polar_from_workbook <- function(
    workbook = "Pct_imp.xlsx",          # or "Norm_pct_imp.xlsx"
    dataset  = "PPMI",                  # "ADNI" / "PPMI"
    cat      = "TP",                    # "TP" / "TN"
    high_val = 6.3,
    xai_cols = c("BP","GBP","LRP","IG","IDGI","OS","RISE","LIME","GC++","SC","LC","OC"),
    out_dir  = img_dir
) {
  wb_path <- file.path(root_dir, workbook)
  sheet_nm <- glue("{dataset}_{cat}")
  
  df <- read_excel(wb_path, sheet = sheet_nm) %>%
    select(label, any_of(xai_cols))
  
  # method display names + ordering
  xai_map <- c(BP="BP", GBP="GBP", IG="IG", LRP="LRP", IDGI="IDGI",
               OS="OS", RISE="RISE", LIME="LIME", GC="GC++", SC="SC", OC="OC", LC="LC")
  xai_levels <- rev(c("BP","GBP","IG","IDGI","LRP","OS","LIME","RISE","GC++","SC","OC","LC"))
  
  df_long <- df %>%
    pivot_longer(-label, names_to = "xai", values_to = "value") %>%
    mutate(
      label = factor(label, levels = unique(label)),
      xai   = factor(recode(xai, !!!xai_map), levels = xai_levels)
    )
  
  # circular x axis with padding + a "gap" column for row labels
  label_levels <- levels(df_long$label)
  gap_levels   <- paste0("gap_", 1:2)
  xai_gap      <- "xai_gap"
  full_levels  <- c(xai_gap, label_levels, gap_levels)
  
  df_long <- df_long %>%
    mutate(label = factor(as.character(label), levels = full_levels))
  
  row_num <- length(levels(df_long$xai))
  
  label_positions <- tibble(
    label = factor(label_levels, levels = label_levels),
    label_id = seq_along(label_levels)
  )
  
  xai_label_positions <- tibble(
    xai = levels(df_long$xai),
    y   = seq_along(levels(df_long$xai)),
    x   = factor(xai_gap, levels = full_levels)
  )
  
  p <- ggplot(df_long, aes(x = label, y = as.numeric(xai), fill = value)) +
    scale_x_discrete(limits = full_levels) +
    ylim(c(-row_num/1.5, row_num + 1.5)) +
    geom_tile(color = "#D3D3D3") +
    scale_fill_gradient(
      limits = c(0, high_val), oob = squish,
      low = "white", high = "#CC0000",
      guide = guide_colorbar(barheight = unit(8, "lines"),
                             barwidth = unit(1.2, "lines"))
    ) +
    geom_text(
      data = label_positions,
      aes(x = label, y = row_num + 1.5, label = label_id),
      inherit.aes = FALSE, size = 4
    ) +
    geom_text(
      data = xai_label_positions,
      aes(x = x, y = y, label = xai),
      inherit.aes = FALSE, size = 3.5, hjust = 1
    ) +
    coord_polar(start = 0) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.text  = element_text(size = 12),
      legend.title = element_text(size = 12),
      plot.margin  = margin(10, 10, 10, 10),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    ) +
    labs(fill = "Percentage")
  
  base <- tools::file_path_sans_ext(basename(workbook))
  out_path <- file.path(out_dir, glue("{base}_{dataset}_{cat}.png"))
  ggsave(out_path, p, width = 8, height = 8, dpi = 300)
  message("Saved: ", out_path)
  p
}

plot_bio_validity_polar_from_workbook("Pct_imp.xlsx", "ADNI", "TP", high_val=12.0)
plot_bio_validity_polar_from_workbook("Pct_imp.xlsx", "ADNI", "TN", high_val=12.0)
plot_bio_validity_polar_from_workbook("Pct_imp.xlsx", "PPMI", "TP", high_val=6.3)
plot_bio_validity_polar_from_workbook("Pct_imp.xlsx", "PPMI", "TN", high_val=6.3)
plot_bio_validity_polar_from_workbook("Norm_pct_imp.xlsx", "ADNI", "TP", high_val=1.0)
plot_bio_validity_polar_from_workbook("Norm_pct_imp.xlsx", "ADNI", "TN", high_val=1.0)
plot_bio_validity_polar_from_workbook("Norm_pct_imp.xlsx", "PPMI", "TP", high_val=1.0)
plot_bio_validity_polar_from_workbook("Norm_pct_imp.xlsx", "PPMI", "TN", high_val=1.0)

# 2. IDV correlation boxplots
plot_idv_box_from_workbook <- function(workbook = "IDV_corrs.xlsx", dataset = "ADNI",
                                       tp_col = "TP", tn_col = "TN") {
  
  wb_path <- file.path(root_dir, workbook)
  
  df <- read_excel(wb_path, sheet = dataset)
  
  df_long <- df %>%
    pivot_longer(cols = all_of(c(tp_col, tn_col)),
                 names_to = "category", values_to = "values") %>%
    mutate(values = map(values, parse_bracket_list)) %>%
    unnest(values) %>%
    mutate(category = factor(category, levels = c(tp_col, tn_col)),
           xai = factor(xai, levels = unique(xai)))
  
  ggplot(df_long, aes(x = xai, y = values, fill = category)) +
    geom_boxplot(position = position_dodge(0.45), width = 0.35,
                 alpha = 0.7, outlier.size = 0.6, outlier.alpha = 0.7) +
    scale_y_continuous(limits = c(-1, 1.1)) +
    scale_fill_manual(values = c("TP" = "#1B9E77", "TN" = "#D95F02")) +
    labs(x = "XAI Method", y = "Score", fill = "Category",
         title = glue("{dataset} IDV correlations")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_idv_box_from_workbook("IDV_corrs.xlsx", "ADNI_all")
plot_idv_box_from_workbook("IDV_corrs.xlsx", "PPMI_all")

# 3. Class correlation tiles
plot_class_corr_tiles_from_workbook <- function(workbook = "Class_corr.xlsx",
                                                sheets = c("ADNI", "PPMI"),
                                                out_dir = img_dir,
                                                out_prefix = "class_corr",
                                                xai_order = NULL) {
  
  wb_path <- file.path(root_dir, workbook)
  
  # read + stack sheets
  df_all <- map_dfr(sheets, \(sh) {
    read_excel(wb_path, sheet = sh) %>% mutate(dataset = sh)
  })
  
  expected <- c("xai","TP_vs_TN","TP_vs_FP","TN_vs_FN","TP_vs_FN","TN_vs_FP","FN_vs_FP","dataset")
  missing_cols <- setdiff(expected, names(df_all))
  if (length(missing_cols) > 0) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  if (is.null(xai_order)) {
    xai_order <- df_all %>% filter(dataset == sheets[1]) %>% distinct(xai) %>% pull(xai)
  }
  
  corr_cols <- setdiff(expected, c("xai","dataset"))
  
  plots <- map(corr_cols, \(col_nm) {
    
    d <- df_all %>%
      transmute(
        dataset = factor(dataset, levels = sheets),
        xai     = factor(xai, levels = xai_order),
        corr    = as.numeric(.data[[col_nm]])
      )
    
    ggplot(d, aes(x = xai, y = dataset, fill = corr)) +
      geom_tile(color = "white", linewidth = 0.3) +
      geom_text(aes(label = sprintf("%.2f", corr)), size = 5) +
      scale_y_discrete(limits = rev(sheets)) +  # ADNI on top
      scale_fill_gradient2(
        low = "#2166AC", mid = "white", high = "#B2182B",
        midpoint = 0, limits = c(-1, 1), oob = squish,
        name = "Correlation"
      ) +
      labs(title = col_nm, x = "XAI", y = "Dataset") +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title  = element_text(face = "bold")
      )
  })
  
  # save each plot
  walk2(plots, corr_cols, \(p, nm) {
    ggsave(
      file.path(out_dir, glue("{out_prefix}_{nm}_ADNI_PPMI.png")),
      p, width = 14, height = 2.8, dpi = 300
    )
  })
  
  plots
}

plot_class_corr_tiles_from_workbook("Class_corrs.xlsx", "ADNI_all")
plot_class_corr_tiles_from_workbook("Class_corrs.xlsx", "PPMI_all")

# 4. Noise robustness plots
plot_noise <- function(path_xlsx,
                       sheet,
                       out_dir = file.path(root_dir, "xai_plots"),
                       out_name = NULL,
                       ylim = c(-0.5, 1.0)) {
  
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  df <- read_excel(path_xlsx, sheet = sheet)
  
  # ---- Wide -> long + assign noise levels LEFT→RIGHT ----
  long_all <- df %>%
    pivot_longer(cols = -xai, names_to = "key", values_to = "vals") %>%
    mutate(
      metric = str_extract(key, "^(TP|TN)"),
      noise_type = key %>%
        str_remove("^(TP|TN)_?") %>%
        str_replace("-.*$", "") %>%
        str_replace("_.*$", "") %>%
        str_to_lower(),
      vals = map(vals, cell_to_numvec)
    ) %>%
    filter(!is.na(metric), !is.na(noise_type)) %>%
    group_by(xai, metric, noise_type) %>%
    mutate(
      # strictly follows column order in Excel
      noise_level = match(key, unique(key))
    ) %>%
    ungroup() %>%
    unnest(vals, keep_empty = FALSE) %>%
    rename(score = vals) %>%
    filter(!is.na(score))
  
  # ---- Summary ----
  sumdat <- long_all %>%
    group_by(xai, noise_type, metric, noise_level) %>%
    summarise(
      n = n(),
      mean = mean(score, na.rm = TRUE),
      sd   = sd(score, na.rm = TRUE),
      se   = sd / sqrt(n),
      .groups = "drop"
    )
  
  # ---- Plot ----
  p <- ggplot() +
    geom_point(
      data = long_all,
      aes(x = noise_level, y = score,
          colour = xai, shape = metric,
          group = interaction(xai, noise_type, metric)),
      position = position_jitter(width = 0.08, height = 0),
      alpha = 0.25, size = 1.6
    ) +
    geom_line(
      data = sumdat,
      aes(x = noise_level, y = mean,
          colour = xai, linetype = noise_type,
          group = interaction(xai, noise_type, metric)),
      linewidth = 0.5
    ) +
    geom_point(
      data = sumdat,
      aes(x = noise_level, y = mean,
          colour = xai, shape = metric,
          group = interaction(xai, noise_type, metric)),
      size = 2.6
    ) +
    scale_x_continuous(
      breaks = sort(unique(sumdat$noise_level)),
      name = "Noise level"
    ) +
    scale_y_continuous(name = "Score") +
    coord_cartesian(ylim = ylim) +
    scale_shape_manual(values = c(TP = 16, TN = 17)) +
    theme_classic() +
    labs(
      title = glue("{sheet}: TP vs TN across noise levels"),
      colour = "XAI method",
      linetype = "Noise type",
      shape = "Group"
    )
  
  if (is.null(out_name)) out_name <- glue("{sheet}.png")
  
  ggsave(
    filename = file.path(out_dir, out_name),
    plot = p,
    width = 12,
    height = 8,
    dpi = 300
  )
  
  message("Saved: ", file.path(out_dir, out_name))
  p
}

plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "ADNI_b",
           file.path(root_dir, "Images"), NULL, c(0.0, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "ADNI_p",
           file.path(root_dir, "Images"), NULL, c(-0.5, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "ADNI_a",
           file.path(root_dir, "Images"), NULL, c(0.0, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "ADNI_r",
           file.path(root_dir, "Images"), NULL, c(-0.5, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "PPMI_b",
           file.path(root_dir, "Images"), NULL, c(-0.25, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "PPMI_p",
           file.path(root_dir, "Images"), NULL, c(-0.5, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "PPMI_a",
           file.path(root_dir, "Images"), NULL, c(0.4, 1.0))
plot_noise(file.path(root_dir, "noise.xlsx"), sheet = "PPMI_r",
           file.path(root_dir, "Images"), NULL, c(-0.5, 1.0))