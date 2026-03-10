local lpeg = require("lpeg")
local P, S, R, C, Cc, Cp, Ct, V = lpeg.P, lpeg.S, lpeg.R, lpeg.C, lpeg.Cc, lpeg.Cp, lpeg.Ct, lpeg.V
local match = lpeg.match
local wrap, yield = coroutine.wrap, coroutine.yield

local ws = S(" \t\n\r") ^ 0
local esc = P("\\") * P(1)
local str_inner = (esc + (1 - S("\"\\")) ^ 1) ^ 0
local jstr = P("\"") * str_inner * P("\"")

local jnum = P("-") ^ -1 *
  (P("0") + R("19") * R("09") ^ 0) *
  (P(".") * R("09") ^ 1) ^ -1 *
  (S("eE") * S("+-") ^ -1 * R("09") ^ 1) ^ -1

local jval = P({
  "val",
  val = ws * (
    P("{") * ws * (V("pair") * (ws * P(",") * ws * V("pair")) ^ 0) ^ -1 * ws * P("}") +
    P("[") * ws * (V("val") * (ws * P(",") * ws * V("val")) ^ 0) ^ -1 * ws * P("]") +
    jstr + jnum + P("true") + P("false") + P("null")
  ) * ws
  ,
  pair = jstr * ws * P(":") * V("val")
})

local key_cap = P("\"") * C(str_inner) * P("\"") * Cp()
local str_pos = P("\"") * Cp() * str_inner * Cp() * P("\"") * Cp()
local val_end = jval * Cp()

local function json_fields(str, fields)
  local fset = {}
  for i = 1, #fields do
    fset[fields[i]] = true
  end
  return wrap(function ()
    local pos = match(ws * P("{") * Cp(), str)
    if not pos then return end
    while true do
      pos = match(ws * Cp(), str, pos)
      if not pos then return end
      local ch = str:sub(pos, pos)
      if ch == "}" then return end
      if ch == "," then
        pos = pos + 1
        pos = match(ws * Cp(), str, pos)
        if not pos then return end
      end
      local key, kend = match(key_cap, str, pos)
      if not key then return end
      pos = match(ws * P(":") * ws * Cp(), str, kend)
      if not pos then return end
      if fset[key] then
        local ch2 = str:sub(pos, pos)
        if ch2 == "\"" then
          local s, e, npos = match(str_pos, str, pos)
          if not s then return end
          if e > s then yield(s, e - 1) end
          pos = npos
        elseif ch2 == "[" then
          pos = pos + 1
          while true do
            pos = match(ws * Cp(), str, pos)
            if not pos then return end
            local ac = str:sub(pos, pos)
            if ac == "]" then pos = pos + 1; break end
            if ac == "," then
              pos = pos + 1
            elseif ac == "\"" then
              local s, e, npos = match(str_pos, str, pos)
              if not s then return end
              if e > s then yield(s, e - 1) end
              pos = npos
            else
              local npos = match(val_end, str, pos)
              if not npos then return end
              pos = npos
            end
          end
        else
          local npos = match(val_end, str, pos)
          if not npos then return end
          pos = npos
        end
      else
        local npos = match(val_end, str, pos)
        if not npos then return end
        pos = npos
      end
    end
  end)
end

local function ci(s)
  local p = P(true)
  for i = 1, #s do
    local c = s:sub(i, i)
    p = p * S(c:lower() .. c:upper())
  end
  return p
end

local squoted = P("'") * (1 - P("'")) ^ 0 * P("'")
local dquoted = P("\"") * (1 - P("\"")) ^ 0 * P("\"")
local tag_body = (squoted + dquoted + (1 - P(">"))) ^ 0 * P(">")
local comment = P("<!--") * (1 - P("-->")) ^ 0 * P("-->")

local function block_elem(name)
  local open = P("<") * ci(name) * #S(" \t\n\r/>") * tag_body
  local close = P("</") * ci(name) * ws * P(">")
  return open * (1 - close) ^ 0 * close
end

local script = block_elem("script")
local style = block_elem("style")
local tag = P("<") * tag_body
local text_span = Cp() * (1 - P("<")) ^ 1 * Cp()

local html_patt = Ct((comment + script + style + tag + text_span) ^ 0)

local function html_text(str)
  local caps = match(html_patt, str)
  local i = 0
  local n = #caps
  return function ()
    while i < n do
      local s = caps[i + 1]
      local e = caps[i + 2]
      i = i + 2
      if e > s then
        return s, e - 1
      end
    end
  end
end

local tag_name_ch = R("az", "AZ") + R("09") + S("-_:")
local tag_name_cap = C(tag_name_ch ^ 1)
local attr_key_patt = (1 - S(" \t\n\r=/>")) ^ 1
local attr_dqv = P("\"") * C((1 - P("\"")) ^ 0) * P("\"")
local attr_sqv = P("'") * C((1 - P("'")) ^ 0) * P("'")
local attr_uqv = C((1 - S(" \t\n\r>\"'")) ^ 1)
local attr_kv = C(attr_key_patt) * ws * P("=") * ws * (attr_dqv + attr_sqv + attr_uqv)
local attr_bare = (1 - S(" \t\n\r=/>\"'")) ^ 1
local attrs_raw = Ct((ws * (attr_kv + attr_bare)) ^ 0)

local close_tag_cap = P("</") * ws * tag_name_cap * ws * P(">") * Cp()
local open_tag_cap = P("<") * tag_name_cap * attrs_raw * ws * (
  P("/") * ws * P(">") * Cp() * Cc(true) +
  P(">") * Cp() * Cc(false)
)

local comment_cp = comment * Cp()
local script_cp = script * Cp()
local style_cp = style * Cp()
local any_tag_cp = P("<") * tag_body * Cp()

local void_elems = {
  area = true, base = true, br = true, col = true, embed = true,
  hr = true, img = true, input = true, link = true, meta = true,
  source = true, track = true, wbr = true,
}

local function html_extract(str)
  local parts = {}
  local tags = {}
  local stack = {}
  local spos = 0
  local pos = 1
  local len = #str
  while pos <= len do
    if str:byte(pos) == 60 then
      local npos = match(comment_cp, str, pos)
        or match(script_cp, str, pos)
        or match(style_cp, str, pos)
      if npos then
        pos = npos
      else
        local cname, cend = match(close_tag_cap, str, pos)
        if cname then
          local lname = cname:lower()
          for i = #stack, 1, -1 do
            if stack[i].lname == lname then
              stack[i].e = spos
              stack[i].close_s = pos
              stack[i].close_e = cend - 1
              stack[i].lname = nil
              tags[#tags + 1] = stack[i]
              table.remove(stack, i)
              break
            end
          end
          pos = cend
        else
          local tname, raw, oend, is_self = match(open_tag_cap, str, pos)
          if tname then
            if not is_self and not void_elems[tname:lower()] then
              local attrs = {}
              for j = 1, #raw, 2 do
                attrs[raw[j]] = raw[j + 1]
              end
              stack[#stack + 1] = {
                name = tname,
                lname = tname:lower(),
                attrs = attrs,
                s = spos + 1,
                open_s = pos,
                open_e = oend - 1,
              }
            end
            pos = oend
          else
            npos = match(any_tag_cp, str, pos)
            if npos then
              pos = npos
            else
              parts[#parts + 1] = "<"
              spos = spos + 1
              pos = pos + 1
            end
          end
        end
      end
    else
      local next_lt = str:find("<", pos, true)
      local text_end = next_lt and (next_lt - 1) or len
      parts[#parts + 1] = str:sub(pos, text_end)
      spos = spos + (text_end - pos + 1)
      pos = text_end + 1
    end
  end
  table.sort(tags, function(a, b) return a.open_s < b.open_s end)
  return table.concat(parts), tags
end

local function html_tags(str)
  local _, tags = html_extract(str)
  local i = 0
  return function ()
    i = i + 1
    local t = tags[i]
    if t then
      return t.name, t.attrs, t.open_s, t.open_e, t.close_s, t.close_e
    end
  end
end

local function html_inject(text, tags)
  local sorted = {}
  for i = 1, #tags do sorted[i] = tags[i] end
  table.sort(sorted, function(a, b) return a.s < b.s end)
  local parts = {}
  local pos = 1
  for i = 1, #sorted do
    local t = sorted[i]
    if t.s > pos then
      parts[#parts + 1] = text:sub(pos, t.s - 1)
    end
    parts[#parts + 1] = "<" .. t.name
    if t.attrs then
      for k, v in pairs(t.attrs) do
        parts[#parts + 1] = " " .. k .. "=\"" .. v:gsub("\"", "&quot;") .. "\""
      end
    end
    parts[#parts + 1] = ">"
    parts[#parts + 1] = t.text or text:sub(t.s, t.e)
    parts[#parts + 1] = "</" .. t.name .. ">"
    pos = t.e + 1
  end
  if pos <= #text then
    parts[#parts + 1] = text:sub(pos)
  end
  return table.concat(parts)
end

return {
  json_fields = json_fields,
  html_text = html_text,
  html_extract = html_extract,
  html_tags = html_tags,
  html_inject = html_inject,
}
