function plot_timebreakdown(saveflag,savedir)

arguments (Input)
    saveflag (1,1) logical = false
    savedir (1,1) string = ""
end

%% files
fileList_f = dir("oz2_results_f_time_NVIDIA_*");
filename_f = strings(length(fileList_f),1);
for i=1:length(fileList_f)
    filename_f(i) = string(fileList_f(i).name);
end
fileList_d = dir("oz2_results_d_time_NVIDIA_*");
filename_d = strings(length(fileList_d),1);
for i=1:length(fileList_d)
    filename_d(i) = string(fileList_d(i).name);
end

Red = "#FF4B00";
Blue = "#005AFF";
Cyan = "#4DC4FF";
Green = "#03AF7A";
Orange = "#F6AA00";
col = [Orange Green Blue Red];
FontSize = 8;

%% float
for fn = 1:length(filename_f)
    filename = filename_f(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    XLIM = [4 12];

    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    labels(contains(labels,"inverse")) = "lines 8–12";
    labels(contains(labels,"GemmEx")) = "line 6";
    labels(contains(labels,"8u")) = "line 7";
    labels(contains(labels,"8i")) = "lines 2–5";

    fig = figure;
    fig.Position(3) = 500;
    fig.Position(4) = 270;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        ax = gca;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        b = bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        for bi = 1:length(b)
            b(bi).FaceColor = col(bi);
        end
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        set(gca,'FontSize',FontSize);
        if ni>1
            yticklabels("");
        else
            ylabel("% (fast)",'FontSize',FontSize+2);
        end
        title("n=" + n_list(ni),'FontSize',FontSize+2);

        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        b = bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        for bi = 1:length(b)
            b(bi).FaceColor = col(bi);
        end
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        set(gca,'FontSize',FontSize);
        if ni>1
            yticklabels("");
        else
            ylabel("% (accurate)",'FontSize',FontSize+2);
        end
    end
    lgd = legend(labels);
    lgd.FontSize = FontSize+2;
    lgd.Layout.Tile = 'north';
    lgd.NumColumns = 4;
    lgd.Direction = "normal";
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"GH200")
        title(lgd, "GH200", 'FontSize',FontSize+2);
    elseif contains(env,"A100")
        title(lgd, "A100 SXM4", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 4090")
        title(lgd, "RTX 4090", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 5080")
        title(lgd, "RTX 5080", 'FontSize',FontSize+2);
    end
    xlabel(t,"Number of moduli",'FontSize',FontSize+2);
    t.TileSpacing = "tight";
    t.Padding = "compact";

    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% double
for fn = 1:length(filename_d)
    filename = filename_d(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 10:13;
    times = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    XLIM = [8 moduli_max];

    labels = replace(string(opts.SelectedVariableNames), "_", "\_");
    labels(contains(labels,"inverse")) = "lines 8–12";
    labels(contains(labels,"GemmEx")) = "line 6";
    labels(contains(labels,"8u")) = "line 7";
    labels(contains(labels,"8i")) = "lines 2–5";

    fig = figure;
    fig.Position(3) = 500;
    fig.Position(4) = 270;
    t = tiledlayout(2,length(n_list));
    for ni = 1:length(n_list)
        nexttile(ni); hold on; grid on;
        ax = gca;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-fast-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_fast = times(flag,:);
        b = bar(OS2_fast./sum(OS2_fast,2)*100,'stacked');
        for bi = 1:length(b)
            b(bi).FaceColor = col(bi);
        end
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        set(gca,'FontSize',FontSize);
        title("n=" + n_list(ni),'FontSize',FontSize+2);
        if ni>1
            yticklabels("");
        else
            ylabel("% (fast)",'FontSize',FontSize+2);
        end

        nexttile(ni + length(n_list)); hold on; grid on;
        flag = false(size(n));
        for m = XLIM(1):XLIM(2)
            flag = flag | contains(func,"OS2-accu-" + m);
        end
        flag = flag & (n == n_list(ni));
        OS2_accu = times(flag,:);
        b = bar(OS2_accu./sum(OS2_accu,2)*100,'stacked');
        for bi = 1:length(b)
            b(bi).FaceColor = col(bi);
        end
        ylim([0 100]);
        xticks(1:(XLIM(2)-XLIM(1)+1));
        xl = string(XLIM(1):XLIM(2));
        xl(2:2:end) = "";
        xticklabels(xl);
        yticks(0:25:100);
        set(gca,'FontSize',FontSize);
        if ni>1
            yticklabels("");
        else
            ylabel("% (accurate)",'FontSize',FontSize+2);
        end
    end
    lgd = legend(labels);
    lgd.FontSize = FontSize+2;
    lgd.Layout.Tile = 'north';
    lgd.NumColumns = 4;
    lgd.Direction = "normal";
    pattern = "time_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"GH200")
        title(lgd, "GH200", 'FontSize',FontSize+2);
    elseif contains(env,"A100")
        title(lgd, "A100 SXM4", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 4090")
        title(lgd, "RTX 4090", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 5080")
        title(lgd, "RTX 5080", 'FontSize',FontSize+2);
    end
    xlabel(t,"Number of moduli",'FontSize',FontSize+2);
    t.TileSpacing = "tight";
    t.Padding = "compact";

    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "time", "timebreakdown");
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

end

