function m = mark(i,j)
markers = {"-", "--", "-.", "-d", "-+", "-o", "-s", "-x", "-p", "-h", "-^", "-v", "->", "-<"};
colors = {"k", "m", "r", "b", "g", "c"};
m = markers{i} + colors(j);
end