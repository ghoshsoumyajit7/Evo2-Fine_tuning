# Code Review

This document describes the process and etiquette for code review in the
BioNeMo repo. You should read this document if you are a developer
working in the BioNeMo repo.

The purpose of these guidelines is to help reduce the friction between
engineers writing code and those reviewing code. As with many rules,
there are exceptions. These exceptions are not comprehensive, so if you
find exceptions that should be listed, please raise them so they can be
evaluated for their inclusion.

## Code Review Process

The code review process is progressive:

1.  Review by your team
2.  Review by domain experts (CODEOWNERS)
3.  Approval by Approval-list users.
4.  (Optional) Coverage Check approval by Approval-list users.

### 1. Team Review

You should first ask contributors to review your change. The contributing team can
provide the most contextualized feedback, and this review step is where
most issues with the change should be caught and addressed.

Proceed to the next step after you've addressed your team's comments and
have received an approval. There is no actual requirement in
gitlab to receive your team-based approval - it is
simply best practice.

### 2. Owner Review

Code owners are domain experts for a particular part of the repository.
They are typically the original authors of a set of source files. Code
ownership structures tend to mirror team structures; a single team is
often responsible for a functional component.

You must receive approval from **at least one** owner of **every** file
you change. Unless the file(s) do not have any owners specified in the `CODEOWNERS` file.

_If your change only modifies files owned by your team, owner review
happens implicitly in the team review step (i.e. in many cases you may
already have this requirement satisfied after step 1 above)._

### 3. Approval

You must also receive approval for your change. Approval indicates that
the change is ready to be merged, and is the final step in the review
process. Self approval is strictly forbidden.

Approval is granted by an approver, in the form of an approval stamp in gitlab.
Approvers review an entire change, in contrast to owners, who
focus on reviewing the files that they own.

To find an approver, first ask your team if there is a preferred
approver. You can also check in #clara-discovery-bionemo-dev.

\*If your team has an approver, approval will usually be granted shortly
after your team has had a chance to review your change.

\*Often, a single review from one person will be sufficient to complete
the entire review process. If the reviewer is on your team, is an
approver, and is an owner of the files you modified, then the process
may collapse to a single review.

### 4. Coverage Check

Our repository has automated checks to ensure test coverage has not regresed.
The coverage check approvers will be the same as the Approval-list users. If
codeline test coverage regresses, the Approvers must make a judgement call
whether it is acceptible or not the merge the code. Occaisonally the coverage
check algorithm has a false positive (i.e. code coverage doesn't regress, yet
coverage check approval is flagged by gitlab), and in this case Approvers are i
free to simply approve the false coverage regression.

## Responsibilities

### All Commenters (Reviewers, Owners, Approvers, etc.)

If a comment thread is start by anyone, it is expected that **the thread starter
resolves the comment**. Resolving a thread by the original thread starter indicates
that the person who started the discussion is happy with the outcome.

### Reviewers

All developers can and should review changes. These reviews are
fundamental to maintaining the quality of the codebase. Reviews are also
an important way to stay aware of what people are working on and
increase the bus factor for areas of the code.

Reviewers should always do the following when reviewing code:

- Be respectful.

- Assume best intentions.

- Review the code, not the person.

- Leave clear, actionable feedback. Use comments in gitlab to signal that you expect
  changes to be made, and explicitly enumerate them. If you don't, it can leave the author of the patch wondering if
  your comments are optional. Requesting code changes should not be interpreted as a
  judgment of the change, but rather as an indicator that the
  reviewer wants to engage with the author to turn it into an
  approval.

A more detailed etiquette guide can be found later in this document.

### Owners

As an owner, you have the ability to delay code from being merged by
withholding your signoff. This ability comes with important
responsibilities:

- A duty to respond to reviews in a timely manner. If you can't stay
  on top of review requests, you should relinquish ownership.

- A bias to "yes". "No", by itself, is **never** an acceptable answer.
  When you reject a change you must work with the change author to
  arrive at a mutually agreeable solution.

### Approvers

As an approver, you have additional responsibilities beyond that of an
owner:

- You are responsible for the change and the effect it has on the
  codebase.

- You must ensure that reviews have been performed by the appropriate
  reviewers, that these reviews were not rushed.

- If a change breaks something, you are expected to be actively
  involved in the cleanup.

You should always decline to approve a change if you're unfamiliar with
the areas of code it touches.

### Becoming an Owner

Code ownership information is stored in CODEOWNERS file. Since these CODEOWNERS file are stored in the
repository, changing them follows the same process as changing code.

To become an owner, add your github username to the CODEOWNERS file that you
want to be a part of, and submit the change to Gitlab. The change to the
CODEOWNERS file will follow the same review process as any other code
change. The existing owners will decide whether to delegate ownership to
you or not.

### Becoming an Approver

As an approver, you are responsible for ensuring the consistency and integrity of our code base.
Before becoming an approver, study this document so that you are completely familiar with the
responsibilities of reviewers and approvers. Additionally, make sure that you are intimately
familiar with our coding style guides and best practices:

- [Contributing](contributing.md)
  - In addition, make sure that you understand and can apply all elements of the
    [Google Python style guide](https://google.github.io/styleguide/pyguide.html), which we adhere
    to for all Python code

## Details on Etiquette for Good Citizens

In the following sections, we provide deeper thoughts and
recommendations for everyone contributing to the repo, in order to have
a fruitful interaction across the team members.

### Patch changes, sizes, and policies

- It takes 30-45 minutes to review every 100-200 lines of code. Be
  mindful of the size of changes you are introducing and try to keep
  your patch under 500 lines of code whenever possible. As a change
  grows in size (lines of codes) the reviewer's ability to find
  issues diminishes.

  - As a corollary, for a refactor-type change, consider separating
    "no-logic-change" and "logic-change" into distinct reviews
    (perhaps in a dependent review change) to ease the pattern
    matching burden on your reviewers.

- All reviewers may request an PR is too large if it is larger than
  500 lines of net code addition. The only exception are MRs into
  the `bionemo2/contrib` directory, where larger MRs are permissible.
  This includes lines of code, but not something such as dummy data or
  a fake dataset that may contain thousands of lines of stuff that is not
  actually functional code.

- Each patch should be kept to one logical change, which should be
  described in the title of the patch. Unrelated changes should be
  split out into separate patches. Fixing whitespace on a line
  you're editing is reasonable. Fixing whitespace around the code
  you're working on should be a separate 'cleanup' patch.

- Where possible, larger patches (\>500 LOC) should be split into
  multiple smaller patches that are consistent individually. Test
  your patches before submitting them to Gitlab via gitlab pipelines or locally.
  The more you test before submitting your patch for review, the better. It's also
  appreciated if you add a line to the commit message describing how
  the patch was tested. This prevents people from having to ask
  whether and how the patch was tested. Stating that the patch was
  not tested is also fine, although you might be asked to do some
  testing in cases where that would be reasonable.

- Abandon patches that are no longer useful or that you don't intend
  to keep working on.

- Follow code styling and rules stated in the project's documents
  (for example, [contributing.md](contributing.md), of which the [Google Python
  Style Guide](https://google.github.io/styleguide/pyguide.html) is a subet) as these define the
  look and feel of the code which defines the most fundamentals of how the code should be
  developed and allows reviewers to focus on the most important aspects of a new piece of code.
  For bash scripting please follow the Google Shell Style Guide [here](https://google.github.io/styleguide/shellguide.html)

- We follow a revert + fix policy in the codebase for any showstopper
  bug that might appear as a result of an PR introducing errors not
  caught by sanity. In exceptional circumstances when an MR cannot be
  reverted and there is a hotfix ready, leadership can consider
  merging without revert. However, this should be an exception.
  Failures policies are described in more depth [here](https://docs.google.com/document/d/1Ryj6sXgzSag6tROrJiKIXcj-l5wTv7jRHJqun-EtNFk/edit#heading=h.xdqlxih4n1fg).

### Review timelines

- In general, patches should remain open for review for at least 24
  hours since the last significant modification to the change. The
  purpose is to let developers around the world have a chance to
  review. Complex reworks, even if they don't change the purpose of
  the patch but the way it's implemented, should restart the waiting
  period.

- Speedy changes: A change can go in without the waiting period if its
  purpose is to fix a recently-introduced issue that has not been
  possible to revert. In that case, the commit message has to
  explain what change introduced the problem and the nature of the
  problem so that the emergency need becomes apparent. The change
  itself should be as limited in scope and impact as possible to
  make it simple to assess the impact.

- Trivial changes that deal with minor issues like inconsistencies in
  whitespace or spelling fixes that don't impact the final binary
  output also don't need to wait for the round of the world reviews.
  Such changes should point out in their commit messages how the
  author verified that the binary output is identical. Note that
  trivial fixes shouldn't necessarily be expedited: Just like
  they're not critical enough for things to go wrong because of
  them, they're not critical enough to require quick handling.

### Reviewers

- Do not approve if you have not reviewed the code you were asked to
  review. Spend time reviewing or re-assign to someone else. Someone
  that approves the change must review the entire change holistically
  If you are a code owner of a particular file, it is appropriate to only reviews the files you own.

- If request an PR change their code, you are responsible for giving concrete
  recommendations for what could be changed to resolve the issue the
  patch addresses. If you feel strongly that a patch should NEVER be
  merged, you are responsible for defending your position and
  listening to other points of view. Asking for changes and walking away is
  not acceptable, and may cause your approval status to be removed by the
  leadership team.

- Include justification for critique: When you review code, you should
  always try to include justification for your critique, unless that
  critique is a nit, a style guide violation, or an obvious bug.
  Nits and style guide violations tend to overlap, and you shouldn't
  have to justify the use of a shared style guide or things like
  proper spelling. Likewise, you shouldn't have to justify bug-free
  code. However, when it comes to design choices you should always
  include justification for when you might want to change certain
  things

- If there have been comments or discussion on a patch, verify that
  the comments have been addressed before giving an approval. If you feel
  that a comment is invalid, please respond to that comment instead
  of just ignoring it.

- Be conscientious when approving patches. As the arbiter, you need to make sure the MR
  is complete, that proper reviews are done, that all the required
  tests have passed, that API changes have been reviewed by the API
  owners. Please make sure that the necessary convergence tests and unit tests
  have passed and have not regressed. If KPI regression (convergence tests) is found, you may need to
  consult with other stakeholders before approving the MR. In some
  cases a Cl will require convergence testing. Owners/ Approvers
  should make sure convergence testing is performed when necessary and
  that no new issues are identified. And that overall, the code
  changes are sound and integrate well with the rest of the modules
  and systems. In the event that the patch breaks things, you are
  expected to be actively involved in the cleanup effort and support
  the authors by reverting and speeding the fixes. This means you
  shouldn't approve a patch just because you trust the author of a
  patch - Make sure you understand what the implications of a patch
  might be, or leave the review to others. Partial reviews,
  reviewing code style, for example, can be given a positive review or a LGTM.
  This also applies if you think the patch looks good, but may
  not have the experience to know if there may be unintended
  consequences.

- Please make sure that the code changes are covered by the existing
  unit tests and if necessary ask the contributor to add or update
  tests.

### Contributors

- Before providing a patch for review ask yourself these questions:

  - Is this PR the right size? If it's too long break it up.

  - Is this MR and all the changes included necessary? (all code has
    to be maintained)

  - Does this MR duplicate existing functionality? If yes, can I
    extend what is there?

  - Is the code readable? Am I using esoteric language constructs
    that affect readability? Does the code follow the conventions
    of the codebase?

  - Is the MR production ready? In other words - does it have tests,
    documentation, error handling, etc, etc.

- Bring attention to patches that you would like reviewed. Add
  reviewers, ask for reviewers on slack or even just rebase it
  against the current codebase to bring it to the top of the Gitlab
  list. If you're not sure who would be a good reviewer, gitlab will suggest OWNERS based
  on the CODEOWNERS file in the UI or look at the
  git history of the files that you've changed, and add those
  people. For NIM-API based changes there is a small team managing these
  therefore seek for those people to review APIs.

- Try to coordinate with other significant contributors to the code
  when making changes to areas you do not own. Before you type a
  single line of code!. These people made the most significant
  changes to that part of the code and therefore are knowledgeable
  of any tradeoffs to be made. Coming with new code already written
  will cause painful back and forth and will be less efficient for
  all. Learn the design, propose changes and get an agreement before
  making changes.

- Don't modify other people's branches unless you have coordinated this
  with the owner of that branch. Not only is this considered rude,
  but your changes could be unintentionally lost. An exception to
  this would be for branches that have not been updated for more than
  90 days and therefore can be considered orphaned.

- Respond to anyone who has taken the time to review your branches,
  even if it's just to say that you disagree. While it may seem
  annoying to address a request to fix spelling or 'trivial' issues,
  it's generally easy to handle in Gitlab's built-in editor. If you
  do use the built-in editor, remember to get that change to your
  local copy before re-pushing. It's also acceptable to add fixes
  for these sorts of comments to another branch, but it's recommended
  that that branch be pushed to Gitlab before the initial branch gets
  submitted.

- Check if there's documentation that needs to be updated to remain
  current after your change. If there's no documentation for the
  part you're working on, consider adding some.

- When contributing a significant change to core parts of the code
  base, or when introducing a new way of doing something that you
  think is worthwhile to apply across the tree, please bring up your
  design doc to the commit, so reviewers can read it.

- Don't expect that people will review your patch unless you ask them
  to. Adding other people as reviewers is the easiest way.
  But also you can use Slack to actively ping the reviewers, which
  is especially useful for urgent MRs.

- Don't expect people to drop all of what they are doing to review
  your patch. Everyone has a day-time job, and while code reviews
  are part of that job, they usually do not extend the full working
  day.

- Do not resolve (ack/Done) any comments open by others so they can
  find their comments easily and resolve them. Unless being told to
  self-resolve.

- As a contributor you are responsible for the code you bring in and
  any failures being caught during unit or convergence testing. Etc.
  You should monitor that your code gets merged successfully and
  that no errors have appeared after the code has been merged
  (nightly testing / convergence testing) and respond promptly to address
  any failures.

### General Etiquette

- We try to assume the best of each other in this community. It's okay
  to discuss mistakes (e.g., isolated instances of non-trivial and
  non-critical changes submitted early) but try to keep such
  inquiries blameless. If a change leads to problems with the code,
  the focus should be on fixing the issue, not on assigning blame.

- Be respectful to others when commenting on branches. Comments should
  be kept to the code and should be kept in a polite tone. Assume
  your colleagues are intelligent and do not intend any malice or
  disrespect. Resist the urge to retaliate against perceived verbal
  misconduct, such behavior is not conducive to getting branches
  merged. Also, avoid absolute and aggressive language as this can
  tend to escalate emotions. Comments are meant to be collaborative.
  Very often, the commenter might also be incorrect since they may not have
  the full story of the patch in mind since they were not the author. A comment
  is not a demand, it's a suggestion towards a mutually acceptable solution
  between the author and the reviewer.

- Don't submit code that you know will break other
  platforms/dependencies. If your patch affects code that is used by
  others, it should be compatible with those. While it would be nice
  to update any other dependents, you must at least provide a path
  that will allow other platforms to continue working.

- Don't write commit messages that are vague or wouldn't make sense to
  partners that read the logs. For example, do not write "\[topic\]
  Bugfix" as your header in the commit message. Keep links to videos
  out of the commit message. Again, partners are going to see these
  logs and it does not make sense to link to something they will not
  have access to view. Keep your commit messages under 72 chars in
  length per line. Good commits have an appropriate topic with a
  nice one-liner that explains the change briefly. This is then
  followed by a few sentences or more on a description of the
  change. Be professional in your language and do not put things in
  the message that a partner would not understand. Avoid the use of
  acronyms, abbreviations, or codenames, especially those meaningful
  only at nvidia. Also, use correct English (check your spelling
  please). This pertains to the final commit message after the branch
  is squashed. Intermediary commit messages do not matter at all. Commits
  must be squashed on merge.

- Consider breaking up large individual patches into smaller patches
  grouped by areas. This makes the patches easier to review but
  increases the number of patches. The way this is handled is a
  personal decision, as long as each patch is still one logical
  change.

- Contributions made to the `bionemo/contrib` folder have more flexible requirements. All requirements not mentioned below still
  apply to the contrib folder. These exceptions are

  - Code line numbers limit is increased to 2200 lines.
  - Unit test coverage requirements are decreased to 65% coverage.
  - Code line length of <=120 character spaces is acceptable.
  - Commit messages do not need have as verbose of an explanation.
    All other requirements pertaining to approver, contributor, and reviewer responsibility still apply.

- Reviews are about the code. It's easy to take it personally when
  someone is criticizing your code, but the whole idea is to get
  better code into our codebase. Again, this also applies in the
  other direction: review code, critically think about and respond
  to the code, but don't make it personal.

- Don't develop on a personal branch for a long time and then dump a
  large number of files and lines of code into a review as a new
  "feature." Features should be developed off of the main trunk just
  like any other change. It is even more important to follow
  processes as the code becomes more critical. Features can
  be broken into small working changes and don't need to be
  developed all at once. Creating large changes like this leads to
  less efficiency in the review process and can lead to more
  mistakes.

- In BioNeMo, for features under development that are not ready for
  production, we put them inside the `bionemo2/contrib` folder. This allows for teams to develop
  faster (less strict reviews) while testing code. When a feature is
  complete and well tested, we move it to `bionemo2/src` or
  `bionemo2/core` and we complete all the requirements for productroduction.

## References

- [Coreboot gerrit guidelines](https://doc.coreboot.org/getting_started/gerrit_guidelines.html) (heavily - inspired the etiquette portion of this document)
- [Code Review Culture](https://engineering.squarespace.com/blog/2019/code-review-culture-part-2)
- [Proven practices for peer review](https://www.ibm.com/developerworks/rational/library/11-proven-practices-for-peer-review/index.html#:~:text=The%20Cisco%20code%20review%20study,a%2070%E2%80%9390%25%20yield)
- [https://confluence.nvidia.com/display/DS/SDK+Code+reviews](https://confluence.nvidia.com/display/DS/SDK+Code+reviews)
- [https://confluence.nvidia.com/display/DS/SDK+Best+Practices](https://confluence.nvidia.com/display/DS/SDK+Best+Practices) (Review and source control sections)
