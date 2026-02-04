function plot_accuracy(saveflag,savedir)

arguments (Input)
    saveflag (1,1) logical = false
    savedir (1,1) string = ""
end

%% files
fileList_f = dir("oz2_results_f_accuracy_NVIDIA_*");
filename_f = strings(length(fileList_f),1);
for i=1:length(fileList_f)
    filename_f(i) = string(fileList_f(i).name);
end
fileList_d = dir("oz2_results_d_accuracy_NVIDIA_*");
filename_d = strings(length(fileList_d),1);
for i=1:length(fileList_d)
    filename_d(i) = string(fileList_d(i).name);
end
FontSize = 8;

if any(contains(filename_f,'GH200')) && any(contains(filename_d,'GH200'))
    fig = figure;
    fig.Position(1) = fig.Position(1) - 100;
    fig.Position(3) = 1000;
    fig.Position(4) = 270;
    t = tiledlayout(2,6);
    k_list = [1024, 16384];

    filename = filename_d(contains(filename_f,'GH200'));
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];
    YLIM = [1e-16,1e6];

    for p = 1:length(phi_list)
        nexttile(p); hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);

        for k=k_list
            DGEMM    = err(phi == phi_list(p) & contains(func,"DGEMM") & contains(func,num2str(k)),:);
            ozIMMU_EF_8 = err(phi == phi_list(p) & contains(func,"ozIMMU_EF-8") & contains(func,num2str(k)),:);
            ozIMMU_EF_9 = err(phi == phi_list(p) & contains(func,"ozIMMU_EF-9") & contains(func,num2str(k)),:);
            OS2_fast = err(phi == phi_list(p) & contains(func,"OS2-fast") & contains(func,num2str(k)),:);
            OS2_accu = err(phi == phi_list(p) & contains(func,"OS2-accu") & contains(func,num2str(k)),:);

            midx = find(k==k_list);
            pl = plot(xx, DGEMM, mark(midx,1), 'DisplayName', "DGEMM (k=" + k +")", 'LineWidth',1);
            li = 0;
            lines = [];
            names = {};
            li=li+1; lines(li) = pl; names{li} = 'DGEMM';
            if ~isempty(ozIMMU_EF_8)
                pl = plot(xx, ozIMMU_EF_8, mark(midx,2), 'DisplayName', "ozIMMU\_EF-8 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'ozIMMU\_EF-8';
            end
            if ~isempty(ozIMMU_EF_9)
                pl = plot(xx, ozIMMU_EF_9, mark(midx,5), 'DisplayName', "ozIMMU\_EF-9 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'ozIMMU\_EF-9';
            end
            pl = plot(xx, OS2_fast, mark(midx,3), 'DisplayName', "OS II-fast (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-fast';
            pl = plot(xx, OS2_accu, mark(midx,4), 'DisplayName', "OS II-accu (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-accu';

            if k==k_list(1)
                LINES = lines;
                NAMES = names;
            end
        end

        ylim(YLIM);
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        if mod(p-1,6) ~= 0
            yticklabels("");
        end
        set(gca,'YScale','Log','FontSize',FontSize);
        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',FontSize+2);
    end
    lgd = legend(LINES, NAMES);
    lgd.Layout.Tile = 6;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
    title(lgd, "GH200", 'FontSize',FontSize+2);

    filename = filename_f(contains(filename_f,'GH200'));
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];
    YLIM = [1e-7,1e6];

    for p = 1:length(phi_list)
        nexttile(p + 6); hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);

        for k=k_list
            SGEMM    = err(phi == phi_list(p) & contains(func,"SGEMM ") & contains(func,num2str(k)),:);
            SGEMM_TF = err(phi == phi_list(p) & contains(func,"SGEMM-TF32") & contains(func,num2str(k)),:);
            SGEMM_BF = err(phi == phi_list(p) & contains(func,"SGEMM-BF16X9") & contains(func,num2str(k)),:);
            FP16TCEC = err(phi == phi_list(p) & contains(func,"FP16TCEC_SCALING") & contains(func,num2str(k)),:);
            OS2_fast = err(phi == phi_list(p) & contains(func,"OS2-fast") & contains(func,num2str(k)),:);
            OS2_accu = err(phi == phi_list(p) & contains(func,"OS2-accu") & contains(func,num2str(k)),:);

            midx = find(k==k_list);
            pl = plot(xx, SGEMM, mark(midx,1), 'DisplayName', "SGEMM (k=" + k +")", 'LineWidth',1);
            li = 0;
            lines = [];
            names = {};
            li=li+1; lines(li) = pl; names{li} = 'SGEMM';
            if ~isempty(SGEMM_TF)
                pl = plot(xx, SGEMM_TF, mark(midx,2), 'DisplayName', "TF32GEMM (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'TF32GEMM';
            end
            if ~isempty(SGEMM_BF)
                pl = plot(xx, SGEMM_BF, mark(midx,5), 'DisplayName', "BF16x9 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'BF16x9';
            end
            if ~isempty(FP16TCEC)
                pl = plot(xx, FP16TCEC, mark(midx,6), 'DisplayName', "cuMpSGEMM (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'cuMpSGEMM';
            end
            pl = plot(xx, OS2_fast, mark(midx,3), 'DisplayName', "OS II-fast (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-fast';
            pl = plot(xx, OS2_accu, mark(midx,4), 'DisplayName', "OS II-accu (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-accu';

            if k==k_list(1)
                LINES = lines;
                NAMES = names;
            end
        end

        ylim(YLIM);
        yticks(10.^(-22:4:30));
        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        if mod(p-1,6) ~= 0
            yticklabels("");
        end
        set(gca,'YScale','Log','FontSize',FontSize);
        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',FontSize+2);
    end
    lgd = legend(LINES, NAMES);
    lgd.Layout.Tile = 12;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
    title(lgd, "GH200", 'FontSize',FontSize+2);

    xlabel(t,"Number of moduli",'FontSize',FontSize+2);
    ylabel(t,"max relative error",'FontSize',FontSize+2);
    t.TileSpacing = "tight";
    t.Padding = "compact";

    if saveflag
        pattern = "(.*?)_NVIDIA";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        figname = replace(figname, "_f_", "_df_");
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% float
for fn = 1:length(filename_f)
    filename = filename_f(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    func_list = unique(func,'stable');
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];
    YLIM = [1e-7,1e6];

    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"RTX")
        k_list = [1024, 12288];
    else
        k_list = [1024, 16384];
    end

    fig = figure;
    fig.Position(1) = fig.Position(1) - 100;
    fig.Position(3) = 1000;
    fig.Position(4) = 150;
    t = tiledlayout(1,6);
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);

        for k=k_list
            SGEMM    = err(phi == phi_list(p) & contains(func,"SGEMM ") & contains(func,num2str(k)),:);
            SGEMM_TF = err(phi == phi_list(p) & contains(func,"SGEMM-TF32") & contains(func,num2str(k)),:);
            SGEMM_BF = err(phi == phi_list(p) & contains(func,"SGEMM-BF16X9") & contains(func,num2str(k)),:);
            FP16TCEC = err(phi == phi_list(p) & contains(func,"FP16TCEC_SCALING") & contains(func,num2str(k)),:);
            OS2_fast = err(phi == phi_list(p) & contains(func,"OS2-fast") & contains(func,num2str(k)),:);
            OS2_accu = err(phi == phi_list(p) & contains(func,"OS2-accu") & contains(func,num2str(k)),:);

            midx = find(k==k_list);
            pl = plot(xx, SGEMM, mark(midx,1), 'DisplayName', "SGEMM (k=" + k +")", 'LineWidth',1);
            li = 0;
            lines = [];
            names = {};
            li=li+1; lines(li) = pl; names{li} = 'SGEMM';
            if ~isempty(SGEMM_TF)
                pl = plot(xx, SGEMM_TF, mark(midx,2), 'DisplayName', "TF32GEMM (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'TF32GEMM';
            end
            if ~isempty(SGEMM_BF)
                pl = plot(xx, SGEMM_BF, mark(midx,5), 'DisplayName', "BF16x9 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'BF16x9';
            end
            if ~isempty(FP16TCEC)
                pl = plot(xx, FP16TCEC, mark(midx,6), 'DisplayName', "cuMpSGEMM (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'cuMpSGEMM';
            end
            pl = plot(xx, OS2_fast, mark(midx,3), 'DisplayName', "OS II-fast (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-fast';
            pl = plot(xx, OS2_accu, mark(midx,4), 'DisplayName', "OS II-accu (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-accu';

            if k==k_list(1)
                LINES = lines;
                NAMES = names;
            end
        end

        ylim(YLIM);
        yticks(10.^(-22:4:30));
        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        if mod(p-1,6) ~= 0
            yticklabels("");
        end
        set(gca,'YScale','Log','FontSize',FontSize);
        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',FontSize+2);
    end
    lgd = legend(LINES, NAMES);
    lgd.Layout.Tile = 6;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
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
    ylabel(t,"max relative error",'FontSize',FontSize+2);
    t.TileSpacing = "tight";
    t.Padding = "compact";

    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

%% double
for fn = 1:length(filename_d)
    filename = filename_d(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 3:length(opts.SelectedVariableNames);
    err = readmatrix(filename,opts);
    err = err(2:end,:);
    opts.SelectedVariableNames = 1;
    phi = readmatrix(filename,opts);
    phi = phi(2:end);
    phi_list = unique(phi,'stable');
    opts.SelectedVariableNames = 2;
    func = string(readmatrix(filename,opts));
    func = func(2:end);
    moduli_min = 2;
    moduli_max = moduli_min + size(err,2) - 1;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];
    YLIM = [1e-16,1e6];

    pattern = "accuracy_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"RTX")
        k_list = [1024, 8192];
    else
        k_list = [1024, 16384];
    end

    fig = figure;
    fig.Position(1) = fig.Position(1) - 100;
    fig.Position(3) = 1000;
    fig.Position(4) = 150;
    t = tiledlayout(1,6);
    for p = 1:length(phi_list)
        nexttile; hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);

        for k=k_list
            DGEMM    = err(phi == phi_list(p) & contains(func,"DGEMM") & contains(func,num2str(k)),:);
            ozIMMU_EF_8 = err(phi == phi_list(p) & contains(func,"ozIMMU_EF-8") & contains(func,num2str(k)),:);
            ozIMMU_EF_9 = err(phi == phi_list(p) & contains(func,"ozIMMU_EF-9") & contains(func,num2str(k)),:);
            OS2_fast = err(phi == phi_list(p) & contains(func,"OS2-fast") & contains(func,num2str(k)),:);
            OS2_accu = err(phi == phi_list(p) & contains(func,"OS2-accu") & contains(func,num2str(k)),:);

            midx = find(k==k_list);
            pl = plot(xx, DGEMM, mark(midx,1), 'DisplayName', "DGEMM (k=" + k +")", 'LineWidth',1);
            li = 0;
            lines = [];
            names = {};
            li=li+1; lines(li) = pl; names{li} = 'DGEMM';
            if ~isempty(ozIMMU_EF_8)
                pl = plot(xx, ozIMMU_EF_8, mark(midx,2), 'DisplayName', "ozIMMU\_EF-8 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'ozIMMU\_EF-8';
            end
            if ~isempty(ozIMMU_EF_9)
                pl = plot(xx, ozIMMU_EF_9, mark(midx,5), 'DisplayName', "ozIMMU\_EF-9 (k=" + k +")", 'LineWidth',1);
                li=li+1; lines(li) = pl; names{li} = 'ozIMMU\_EF-9';
            end
            pl = plot(xx, OS2_fast, mark(midx,3), 'DisplayName', "OS II-fast (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-fast';
            pl = plot(xx, OS2_accu, mark(midx,4), 'DisplayName', "OS II-accu (k=" + k +")", 'LineWidth',1);
            li=li+1; lines(li) = pl; names{li} = 'OS II-accu';

            if k==k_list(1)
                LINES = lines;
                NAMES = names;
            end
        end

        ylim(YLIM);
        yticks(10.^(-20:4:30));
        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        if mod(p-1,6) ~= 0
            yticklabels("");
        end
        set(gca,'YScale','Log','FontSize',FontSize);
        title("$\phi=" + phi_list(p) + "$",'Interpreter','Latex','FontSize',FontSize+2);
    end
    lgd = legend(LINES, NAMES);
    lgd.Layout.Tile = 6;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
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
    ylabel(t,"max relative error",'FontSize',FontSize+2);
    t.TileSpacing = "tight";
    t.Padding = "compact";

    if saveflag
        pattern = "(.*?)_2025";
        match = regexp(filename, pattern, 'tokens');
        figname = match{1}{1};
        savefig(fig,savedir+figname);
        exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
    end
end

end

