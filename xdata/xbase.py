# XML/XSLT Parser module
# ----------------------
# Showing the steps to transform XML using XSL stylesheet

# The main package
import lxml.etree as et

# Open and parse xml doc
dom = et.parse("xml/xsystem.smp.xml")

# Open and parse xsl doc
xsl = et.parse("xsl/jsitemap.xsl")

# create xslt processor
xslt = et.XSLT(xsl)

# transform doc into variable
result = xslt(dom)

# cast result as string
str(result)

# output result
result.write_output("html/sitemap.html")


# get root
result.getroot().text
