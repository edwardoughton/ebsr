#Visualize EBSR data
library(tidyverse)
library(ggpubr)

#get folder directory
folder <- dirname(rstudioapi::getSourceEditorContext()$path)
path = file.path(folder, '..', 'results', "country_results.csv")
data = read.csv(path)

data$cost_per_sp_user_using_existing_infra[data$cost_per_sp_user_using_existing_infra > 1000] <- 1000

data$income = factor(data$income, levels=c("hic",
                                               'umc',
                                               "lmc", 
                                           'lic'
),
labels=c("High Income Country",
         'Upper Middle Income Country',
         "Lower Middle Income Country", 
         'Low Income Country'
))

###################################################
subset = select(
  data, 
  income, 
  continent,
  smartphone_users_perc, 
  cost_per_sp_user_using_existing_infra,
)

plot1 = ggplot(subset,
  aes(x=factor(smartphone_users_perc),
      y=cost_per_sp_user_using_existing_infra,
      fill=income)) +
  geom_boxplot() + expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(20, 400)) + facet_wrap(~income, ncol=2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  labs(title = "(A) Decile Cost Per Smartphone Subscriber by Income Group",
       subtitle = "Based on reusing as much existing available infrastructure as possible.",
       x = "Population Decile (%)",
       y = "Cost Per Smartphone User (US$)")

###
subset = select(
  data,
  income,
  continent,
  smartphone_users_perc,
  cost_per_sp_user_entirely_gf,
)

plot2 = ggplot(subset,
               aes(x=factor(smartphone_users_perc),
                   y=cost_per_sp_user_entirely_gf,
                   fill=income)) +
  geom_boxplot() + expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(20, 400)) + facet_wrap(~income, ncol=2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  labs(title = "(B) Decile Cost Per Smartphone Subscriber by Income Group",
       subtitle = "Based on building from scratch all greenfield infrastructure.",
       x = "Population Decile (%)",
       y = "Cost Per Smartphone User (US$)")

panel <- ggarrange(plot1, plot2, 
                   ncol = 1, nrow = 2, align = c("hv"), 
                   common.legend=T, legend='bottom')

path = file.path(folder, 'figures', 'panel_cost_per_user_by_income.png')
ggsave(path, units="in", width=8, height=10, dpi=300)
print(panel)
dev.off()

###################################################
subset = select(
  data,
  income,
  continent,
  smartphone_users_perc,
  cost_per_sp_user_using_existing_infra,
)

plot3 = ggplot(subset,
               aes(x=factor(smartphone_users_perc),
                   y=cost_per_sp_user_using_existing_infra,
                   fill=continent)) +
  geom_boxplot() + expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(20, 400)) + facet_wrap(~continent, ncol=2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(fill=guide_legend(ncol=6)) +
  labs(title = "(A) Decile Cost Per Smartphone Subscriber by Region",
       subtitle = "Based on reusing as much existing available infrastructure as possible.",
       x = "Population Decile (%)",
       y = "Cost Per Smartphone User (US$)")

###
subset = select(
  data,
  income,
  continent,
  smartphone_users_perc,
  cost_per_sp_user_entirely_gf,
)

plot4 = ggplot(subset,
               aes(x=factor(smartphone_users_perc),
                   y=cost_per_sp_user_entirely_gf,
                   fill=continent)) +
  geom_boxplot() + expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(20, 400)) + facet_wrap(~continent, ncol=2) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(fill=guide_legend(ncol=6)) +
  labs(title = "(B) Decile Cost Per Smartphone Subscriber by Region",
       subtitle = "Based on building from scratch all greenfield infrastructure.",
       x = "Population Decile (%)",
       y = "Cost Per Smartphone User (US$)")

panel2 <- ggarrange(plot3, plot4, 
                   ncol = 1, nrow = 2, align = c("hv"), 
                   common.legend=T, legend='bottom')

path = file.path(folder, 'figures', 'panel_cost_per_user_by_region.png')
ggsave(path, units="in", width=8, height=10, dpi=300)
print(panel2)
dev.off()

###################################################
subset = select(
  data,
  income,
  # continent,
  smartphone_users_perc,
  total_cost_using_existing_infra_tco,
)

subset = subset %>%
  group_by(smartphone_users_perc, income) %>%
  summarize(
    total_cost_using_existing_infra_tco = sum(total_cost_using_existing_infra_tco)
  )

plot5 = ggplot(subset, aes(x=factor(smartphone_users_perc),
                   y=(total_cost_using_existing_infra_tco/1e9),
                   group=income)) +
  geom_line(aes(color=income)) + 
  expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(0, 600)) +
  # facet_wrap(~continent, ncol=3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(fill=guide_legend(ncol=6)) +
  labs(title = "(A) Cumulative Cost Income Group",
       subtitle = "Based on infrastructure reuse.",
       x = "Population Decile (%)",
       y = "Cumulative Cost (US$)")

###
subset = select(
  data,
  income,
  continent,
  smartphone_users_perc,
  total_cost_entirely_greenfield_tco,
)

subset = subset %>%
  group_by(smartphone_users_perc, income) %>%
  summarize(
    total_cost_entirely_greenfield_tco = sum(total_cost_entirely_greenfield_tco)
  )

plot6 = ggplot(subset, aes(x=factor(smartphone_users_perc),
                   y=(total_cost_entirely_greenfield_tco/1e9),
                   group=income)) +
  geom_line(aes(color=income)) + 
  expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(0, 600)) +
  # facet_wrap(~continent, ncol=3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(fill=guide_legend(ncol=6)) +
  labs(title = "(B) Cumulative Cost Income Group",
       subtitle = "Based on a greenfield approach",
       x = "Population Decile (%)",
       y = "Cumulative Cost (US$)")


####
subset = select(
  data,
  # income,
  continent,
  smartphone_users_perc,
  total_cost_using_existing_infra_tco,
)

subset = subset %>%
  group_by(smartphone_users_perc, continent) %>%
  summarize(
    total_cost_using_existing_infra_tco = sum(total_cost_using_existing_infra_tco)
  )

plot7 = ggplot(subset, aes(x=factor(smartphone_users_perc),
                           y=(total_cost_using_existing_infra_tco/1e9),
                           group=continent)) +
  geom_line(aes(color=continent)) + 
  expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(0, 590)) +
  # facet_wrap(~continent, ncol=3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(color=guide_legend(ncol=6)) +
  labs(title = "(C) Cumulative Cost by Region",
       subtitle = "Based on infrastructure reuse.",
       x = "Population Decile (%)",
       y = "Cumulative Cost (US$)")

####
subset = select(
  data,
  # income,
  continent,
  smartphone_users_perc,
  total_cost_entirely_greenfield_tco,
)

subset = subset %>%
  group_by(smartphone_users_perc, continent) %>%
  summarize(
    total_cost_entirely_greenfield_tco = sum(total_cost_entirely_greenfield_tco)
  )

plot8 = ggplot(subset, aes(x=factor(smartphone_users_perc),
                           y=(total_cost_entirely_greenfield_tco/1e9),
                           group=continent)) +
  geom_line(aes(color=continent)) + 
  expand_limits(x = 5, y = 5) +
  scale_y_continuous(limits = c(0, 590)) +
  # facet_wrap(~continent, ncol=3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom", legend.title=element_blank()) +
  guides(color=guide_legend(ncol=6)) +
  labs(title = "(D) Cumulative Cost by Region",
       subtitle = "Based on a greenfield approach.",
       x = "Population Decile (%)",
       y = "Cumulative Cost (US$)")

step1 <- ggarrange(plot5, plot6,
                    ncol = 2, nrow = 1, align = c("hv"),
                    common.legend=T, legend='bottom')

step2 <- ggarrange(plot7, plot8,
                   ncol = 2, nrow = 1, align = c("hv"),
                   common.legend=T, legend='bottom')


panel2 <- ggarrange(step1, step2,
                    ncol = 1, nrow = 2, align = c("hv"),
                    common.legend=F, legend='bottom')

path = file.path(folder, 'figures', 'panel_cumulative_cost_by_region.png')
ggsave(path, units="in", width=8, height=6, dpi=300)
print(panel2)
dev.off()

